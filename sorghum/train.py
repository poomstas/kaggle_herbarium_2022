# %%
import os
import csv
import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from torchsummary import summary

from src.data import SorghumDataset
from src.constants import CULTIVAR_LABELS_IND2STR, IMAGENET_NORMAL_MEAN, IMAGENET_NORMAL_STD, BACKBONE_IMG_SIZE
from src.utils import balance_val_split, get_stratified_sampler_for_subset
from src.learnable_resizer import LearnToResize, LearnToResizeKaggle

# %% Hyperparameters
RESIZER             = False         # Apply "Learning to Resize Images for Computer Vision Tasks by Talebi et al., (2021)"
PRETRAINED          = True
N_HIDDEN_NODES      = 2048          # No hidden layer if None, backbone out has 2048, final has 100
DROPOUT_RATE        = 0.3           # No dropout if 0
NUM_CLASSES         = 100           # Fixed (for this challenge)
NUM_EPOCHS          = 60    
LR                  = 0.0001
NUM_WORKERS         = os.cpu_count()
BACKBONE            = 'efficientnet-b7' # ['xception', 'efficientnet-b3', 'efficientnet-b7', 'resnest-269']
FREEZE_BACKBONE     = False
UNFREEZE_AT         = 99999         # Disables freezing if 0 (epoch count starts at 0)

host_name = socket.gethostname()
print('\nHost Name: {}\n'.format(host_name))

# Effective batch size = batch_size * gpus * num_nodes. 256 on A100, 64 on GTX 1080Ti
if BACKBONE == 'xception':
    if host_name=='jupyter-brian':
        BATCH_SIZE = 30
    elif host_name=='hades-ubuntu':
        BATCH_SIZE = 32
    else:
        BATCH_SIZE = 256
elif BACKBONE == 'efficientnet-b3':
    if host_name=='jupyter-brian':
        BATCH_SIZE = 99999
    elif host_name=='hades-ubuntu':
        BATCH_SIZE = 99999
    else:
        BATCH_SIZE = 32 if RESIZER else 16
elif BACKBONE == 'efficientnet-b7':
    if host_name=='jupyter-brian':
        BATCH_SIZE = 99999
    elif host_name=='hades-ubuntu':
        BATCH_SIZE = 99999
    else:
        BATCH_SIZE = 32 if RESIZER else 6
elif BACKBONE == 'resnest-269':
    if host_name=='jupyter-brian':
        BATCH_SIZE = 99999
    elif host_name=='hades-ubuntu':
        BATCH_SIZE = 99999
    else:
        BATCH_SIZE = 32 if RESIZER else 64

transform_list_train = [
        # A.RandomResizedCrop(height=BACKBONE_IMG_SIZE[BACKBONE], width=BACKBONE_IMG_SIZE[BACKBONE]), # Improved final score by 0.023 (0.575->0.598)
        A.HorizontalFlip(p=0.5), # Leaving this on improved performance (at 0.5)
        A.VerticalFlip(p=0.5), # Leaving this on improved performance (at 0.5)
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.OneOf([ # Including this improved performance from 0.725 to 0.730
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
        ], p=0.5),
        A.OneOf([ # Decreased performance from 0.730 to 0.723
            A.Blur(p=0.1),
            A.GaussianBlur(p=0.1),
            A.MotionBlur(p=0.1),
        ], p=0.1),
        A.OneOf([ # Increased performance from 0.723 to 0.727
            A.GaussNoise(p=0.1),
            A.ISONoise(p=0.1),
            A.GridDropout(ratio=0.5, p=0.2),
            A.CoarseDropout(max_holes=16, min_holes=8, max_height=16, max_width=16, min_height=8, min_width=8, p=0.2)
        ], p=0.2),
        A.Normalize(IMAGENET_NORMAL_MEAN, IMAGENET_NORMAL_STD),
        ToTensorV2(), # np.array HWC image -> torch.Tensor CHW]
]
trainsform_list_val  = [
        # A.Resize(height=BACKBONE_IMG_SIZE[BACKBONE], width=BACKBONE_IMG_SIZE[BACKBONE]),
        A.Normalize(IMAGENET_NORMAL_MEAN, IMAGENET_NORMAL_STD),
        ToTensorV2(), # np.array HWC image -> torch.Tensor CHW]
]

if RESIZER: # Remove the resizing augmentations
    transform_list_train.pop(0)
    trainsform_list_val.pop(0)

TRANSFORMS = {'train': A.Compose(transform_list_train), 'val': A.Compose(trainsform_list_val)}

# Save TRANSFORMS to YAML (Use as example for better organizing runs)
A.save(TRANSFORMS['train'], 'transform_train.yml', data_format='yaml')
TRANSFORMS['train'] = A.load('transform_train.yml', data_format='yaml') # How to load

# Transforms above inspired by the following post:
#   https://www.kaggle.com/code/pegasos/sorghum-pytorch-lightning-starter-training

TB_NOTES = "InputRes1024_3FC_ReducedMaxLR_BaseCase"
TB_NOTES += "_" + host_name + "_" + BACKBONE + "_" + str(N_HIDDEN_NODES) + "_UnfreezeAt" + str(UNFREEZE_AT) + "_ResizerApplied_" + str(RESIZER)


# %%
class SorghumLitModel(pl.LightningModule):
    def __init__(self, num_epochs, backbone, transforms, num_classes, batch_size, lr, n_hidden_nodes, 
                 dropout_rate=0, pretrained=True, num_workers=4, freeze_backbone=False, unfreeze_at=0):
        super(SorghumLitModel, self).__init__()
        self.save_hyperparameters() # Need this later to load_from_checkpoint without providing the hyperparams again

        self.num_epochs = num_epochs
        self.transforms = transforms # {'train': A.Compose([...]), 'val': A.Compose([...])}
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers= num_workers
        self.n_hidden_nodes = n_hidden_nodes
        self.freeze_backbone = freeze_backbone
        self.unfreeze_at = unfreeze_at

        self.now = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.tests_result_csv_filename = 'test_result_{}.csv'.format(self.now)
        self.csv_header_written = False
        print('Run ID: ', self.now)

        if backbone == 'xception': # backbone_input_size = 3 x 299 x 299 (fixed)
            from src.model import XceptionModel
            self.model = XceptionModel(num_classes, pretrained, n_hidden_nodes, dropout_rate, freeze_backbone)
        elif backbone == 'efficientnet-b3': # backbone_input_size can be adjusted, but we'll set it to 3 x 350 x 350
            from src.model import EfficientNetModel
            self.model = EfficientNetModel(n_hidden_nodes, dropout_rate, freeze_backbone, version='b3')
        elif backbone == 'efficientnet-b7': # backbone_input_size can be adjusted, but we'll set it to 3 x 350 x 350
            from src.model import EfficientNetModel
            self.model = EfficientNetModel(n_hidden_nodes, dropout_rate, freeze_backbone, version='b7')
        elif backbone == 'resnest-269':
            from src.model import ResNeSt269
            self.model = ResNeSt269(n_hidden_nodes, dropout_rate, freeze_backbone)

        if RESIZER:
            target_size_xy = (BACKBONE_IMG_SIZE[backbone], BACKBONE_IMG_SIZE[backbone])
            resizer_module = LearnToResize(num_res_blocks=1, target_size=target_size_xy)
            self.model = nn.Sequential(resizer_module, self.model)

        summary(model       = self.model, 
                input_size  = (3, BACKBONE_IMG_SIZE[backbone], BACKBONE_IMG_SIZE[backbone]), 
                batch_size  = self.batch_size,
                device      = 'cpu')

    def forward(self, x):
        return self.model.forward(x)

    def setup(self, stage=None):
        '''
        The setup hook is used for the following:
            - count number of classes
            - build vocabulary
            - perform train/val/test splits
            - create datasets
            - apply transforms (defined explicitly in your datamodule)
            - etc...

        Called at the beginning of fit (train + validate), validate, test, or predict. This is a good hook when
        you need to build models dynamically or adjust something about them. This hook is called on every process
        when using DDP.

        stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``
        '''

        if stage in (None, "fit"):
        # ==================================================================================================
        #       Apply separate transforms for train and val sets
        #       Separate batch samplers for train and val (stratified sampling + DDP)
        #       Reference:
        #         https://discuss.pytorch.org/t/changing-transforms-after-creating-a-dataset/64929/5
        # ==================================================================================================

            csv_fullpath = './data/sorghum/train_cultivar_mapping.csv'
            assert os.path.exists(csv_fullpath), '.csv file does not exist. Check directory reference.'

            train_dataset = SorghumDataset(csv_fullpath     = csv_fullpath,
                                           transform        = self.transforms['train'],
                                           testset          = False)
            val_dataset   = SorghumDataset(csv_fullpath     = csv_fullpath,
                                           transform        = self.transforms['val'],
                                           testset          = False)

            # Stratified separation of training and validation sets
            train_indx, val_indx = balance_val_split(dataset        = train_dataset,
                                                     stratify_by    = 'cultivar_indx',
                                                     test_size      = 0.2,
                                                     random_state   = None)

            # Get subsets from separate dataset sources bec. transforms are different
            self.train_dataset = Subset(train_dataset, indices=train_indx)
            self.val_dataset   = Subset(val_dataset  , indices=val_indx)

            # Stratified sampler for generating batches (within training and validation stages)
            self.train_sampler, _ = get_stratified_sampler_for_subset(self.train_dataset, target_variable_name='cultivar_indx')
            self.val_sampler = None     # Don't apply stratified sampling within validation sets

            shuffle = False if self.train_sampler is not None else True

            self.train_loader = DataLoader(dataset       = self.train_dataset,
                                           batch_size    = self.batch_size, 
                                           shuffle       = shuffle, 
                                           num_workers   = self.num_workers,
                                           persistent_workers = True,
                                           sampler       = self.train_sampler)

            self.val_loader = DataLoader(dataset         = self.val_dataset, 
                                         batch_size      = self.batch_size, 
                                         shuffle         = False, 
                                         num_workers     = self.num_workers,
                                         persistent_workers = True,
                                         sampler         = self.val_sampler)

        if stage in (None, "test"):
            pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                        epochs              = self.num_epochs, 
                                                        steps_per_epoch     = len(self.train_loader), # The number of steps per epoch to train for. This is used along with epochs in order to infer the total number of steps in the cycle if a value for total_steps is not provided. Default: None
                                                        max_lr              = 2.5e-4/4, 
                                                        pct_start           = 0.3,  # The percentage of the cycle spent increasing the learning rate Default: 0.3
                                                        div_factor          = 25,   # Determines the initial learning rate via initial_lr = max_lr/div_factor Default: 25
                                                        final_div_factor    = 5e+4) # Determines the minimum learning rate via min_lr = initial_lr/final_div_factor Default: 1e4
        scheduler = {'scheduler': scheduler, 'interval': 'step'}

        return [optimizer], [scheduler]

        '''
        Alternative #1:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer = optimizer, 
                                      patience  = 4, 
                                      factor    = 0.3, 
                                      min_lr    = 0.00001, 
                                      verbose   = True)
        return [optimizer], [scheduler]
        
        Alternative #2:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [scheduler]
        '''

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def training_step(self, batch, batch_idx):
        images, cultivar_indx = batch
        pred = self.forward(images) # Forward pass
        train_loss = F.cross_entropy(pred, cultivar_indx)

        correct = pred.argmax(dim=1).eq(cultivar_indx).sum().item()
        accuracy = correct / len(cultivar_indx)
        self.log('train_loss', train_loss)
        self.log('train_acc', accuracy)

        return {'loss': train_loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss_avg', avg_loss)
        
    def validation_step(self, batch, batch_idx):
        images, cultivar_indx = batch
        pred = self.forward(images) # Forward pass
        val_loss = F.cross_entropy(pred, cultivar_indx)

        correct = pred.argmax(dim=1).eq(cultivar_indx).sum().item()
        accuracy = correct / len(cultivar_indx)
        self.log('val_loss', val_loss)
        self.log('val_acc', accuracy)

        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs): # Run this at the end of a validation epoch
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss_avg', avg_loss)

    def on_train_epoch_start(self):
        if self.current_epoch == self.unfreeze_at:
            print('Unfreezing Backbone...')
            for param in self.model.backbone.parameters():
                param.requires_grad = True
            print('Backbone unfreezing complete.')

    def test_step(self, batch, batch_idx):
        images, filenames = batch
        filenames = list(filenames)
        out = self.forward(images)
        out_indx = torch.argmax(out, dim=1).tolist()

        with open(self.tests_result_csv_filename, 'a') as f:
            writer = csv.writer(f)
            if not self.csv_header_written:
                self.csv_header_written = True
                writer.writerow(['filename', 'cultivar'])
            for classification, filename in zip(out_indx, filenames):
                writer.writerow([filename, CULTIVAR_LABELS_IND2STR[classification]])

# %%
if __name__=='__main__':
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    if TB_NOTES != '':
        now += '_' + TB_NOTES

    logger_tb = TensorBoardLogger('./tb_logs', name=now)
    logger_wandb = WandbLogger(project='Sorghum', name=now, mode='online') # online or disabled

    # Saves checkpoints at every epoch
    cb_checkpoint = ModelCheckpoint(dirpath     = './tb_logs/{}/'.format(now), 
                                    monitor     = 'val_loss', 
                                    filename    = '{epoch:02d}-{val_loss:.5f}',
                                    save_top_k  = 3)

    cb_earlystopping = EarlyStopping(monitor    = 'val_loss',
                                     patience   = 8,
                                     strict     = True, # whether to crash the training if monitor is not found in the val metrics
                                     verbose    = True,
                                     mode       = 'min')

    cb_lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(max_epochs            = NUM_EPOCHS, 
                      fast_dev_run          = False,     # Run a single-batch through train and val and see if the code works. No TB or wandb logs
                      gpus                  = -1,        # -1 to use all available GPUs, [0, 1, 2] to specify GPUs by index
                      auto_select_gpus      = True,
                      auto_lr_find          = True,
                      default_root_dir      = '../', 
                      precision             = 16,  # mixed precision training
                      logger                = [logger_tb, logger_wandb],
                      log_every_n_steps     = 10,
                      accelerator           = 'ddp',
                      callbacks             = [cb_checkpoint, cb_earlystopping, cb_lr_monitor],
                      plugins               = DDPPlugin(find_unused_parameters=False),
                      replace_sampler_ddp   = False) # False when using custom sampler

    model = SorghumLitModel(num_epochs      = NUM_EPOCHS,
                            backbone        = BACKBONE, 
                            transforms      = TRANSFORMS, 
                            num_classes     = NUM_CLASSES,
                            batch_size      = BATCH_SIZE, 
                            lr              = LR, 
                            pretrained      = PRETRAINED, 
                            n_hidden_nodes  = N_HIDDEN_NODES, 
                            dropout_rate    = DROPOUT_RATE,
                            num_workers     = NUM_WORKERS, 
                            freeze_backbone = FREEZE_BACKBONE,
                            unfreeze_at     = UNFREEZE_AT)

    trainer.fit(model)

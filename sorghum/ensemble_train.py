# %%
import os
import csv
import socket
from matplotlib import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from datetime import datetime
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from src.data import SorghumDataset
from src.constants import CULTIVAR_LABELS_IND2STR, IMAGENET_NORMAL_MEAN, IMAGENET_NORMAL_STD, BACKBONE_IMG_SIZE
from src.utils import balance_val_split, get_stratified_sampler_for_subset
from src.learnable_resizer import LearnToResize, LearnToResizeKaggle
from src.model import Identity

from train import SorghumLitModel

# %%
transform_list_train = [
        # A.RandomResizedCrop(height=BACKBONE_IMG_SIZE[BACKBONE], width=BACKBONE_IMG_SIZE[BACKBONE]),
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

TRANSFORMS = {'train': A.Compose(transform_list_train), 'val': A.Compose(trainsform_list_val)}

# %%
# %% Hyperparameters
DROPOUT_RATE        = 0.5           # No dropout if 0
NUM_EPOCHS          = 60    
LR                  = 0.0001
NUM_WORKERS         = os.cpu_count() // 2
BATCH_SIZE          = 42 # 43 failed OOM; 40 reached 33671 MB; 42 reached 35147 MB

host_name = socket.gethostname()
print('\nHost Name: {}\n'.format(host_name))

now = datetime.now().strftime('%Y%m%d_%H%M%S')
TB_NOTES = "Ensemble_{}".format(now)
TB_NOTES += "_" + host_name
TB_NOTES += "_DropoutRate{}".format(str(DROPOUT_RATE))

# %%
class EnsembleModel(pl.LightningModule):
    def __init__(self, model_1_path, model_2_path, model_3_path, num_epochs, transforms, batch_size, lr, num_workers, dropout_rate=0.3):
        super(EnsembleModel, self).__init__()
        self.save_hyperparameters() # Need this later to load_from_checkpoint without providing the hyperparams again

        self.num_epochs      = num_epochs
        self.transforms      = transforms
        self.batch_size      = batch_size
        self.lr              = lr
        self.num_workers     = num_workers
        
        # Save the model paths in case of future reference
        self.model_1_path   = model_1_path
        self.model_2_path   = model_2_path
        self.model_3_path   = model_3_path

        # Load the pre-trained models
        self.model_1         = SorghumLitModel.load_from_checkpoint(checkpoint_path=model_1_path)
        self.model_2         = SorghumLitModel.load_from_checkpoint(checkpoint_path=model_2_path)
        self.model_3         = SorghumLitModel.load_from_checkpoint(checkpoint_path=model_3_path)

        # Replace the last unnecessary layer with an Identity layer (alternatively, use Identity() from src.model)
        self.model_1.model.fc3 = torch.nn.Identity()
        self.model_2.model.fc3 = torch.nn.Identity()
        self.model_3.model.fc3 = torch.nn.Identity()

        # Freeze pre-trained parts of the models
        for param in self.model_1.parameters():
            param.requires_grad = False
        for param in self.model_2.parameters():
            param.requires_grad = False
        for param in self.model_3.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_A = nn.LazyLinear(512) # 1536 to 512
        self.fc_B = nn.LazyLinear(256) # 500 to 256
        self.fc_C = nn.LazyLinear(100) # 256 to 100
        self.relu_A = nn.ReLU()
        self.relu_B = nn.ReLU()
        self.relu_C = nn.ReLU()

        self.csv_header_written = False
        self.now = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.tests_result_csv_filename = 'test_result_{}.csv'.format(self.now)

    # Frankenstein the models together
    def forward(self, x):
        out1, out2, out3 = self.model_1(x), self.model_2(x), self.model_3(x)
        out = torch.cat((out1, out2, out3), 1) # 512 nodes each, 1536 nodes total after concat
        out = self.dropout(out)
        out = self.fc_A(out)
        out = self.relu_A(out)
        out = self.dropout(out)
        out = self.fc_B(out)
        out = self.relu_B(out)
        out = self.dropout(out)
        out = self.fc_C(out) # No activation and no softmax at the end (contained in F.cross_entropy())

        return out

    def setup(self, stage=None):
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                        epochs              = self.num_epochs, 
                                                        steps_per_epoch     = len(self.train_loader), # The number of steps per epoch to train for. This is used along with epochs in order to infer the total number of steps in the cycle if a value for total_steps is not provided. Default: None
                                                        max_lr              = 2.5e-4/8, 
                                                        pct_start           = 0.3,  # The percentage of the cycle spent increasing the learning rate Default: 0.3
                                                        div_factor          = 25,   # Determines the initial learning rate via initial_lr = max_lr/div_factor Default: 25
                                                        final_div_factor    = 5e+4) # Determines the minimum learning rate via min_lr = initial_lr/final_div_factor Default: 1e4
        scheduler = {'scheduler': scheduler, 'interval': 'step'}

        return [optimizer], [scheduler]

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
    MODEL_1_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220414_140928_InputRes1024_3FC_ReducedMaxLR_BaseCase_nipa2022-49703_efficientnet-b3_2048_UnfreezeAt99999_ResizerApplied_False/epoch=48-val_loss=0.00391.ckpt" # 0.858
    MODEL_2_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220424_073255_InputRes1024_3FC_ReducedMaxLR_BaseCase_nipa2022-49703_efficientnet-b7_2048_UnfreezeAt99999_ResizerApplied_False/epoch=35-val_loss=0.00229.ckpt" # 0.852
    MODEL_3_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220420_035255_InputRes1024_3FC_ReducedMaxLR_nipa2022-49703_efficientnet-b3_2048_UnfreezeAt99999_ResizerApplied_False_DropoutRate0.5_BaseCase_20220414_140928/epoch=53-val_loss=0.00167.ckpt" # 0.849

    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    if TB_NOTES != '':
        now += '_' + TB_NOTES

    logger_tb = TensorBoardLogger('./tb_logs', name=now)
    logger_wandb = WandbLogger(project='SorghumEnsemble', name=now, mode='online') # online or disabled

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

    model = EnsembleModel(model_1_path  = MODEL_1_PATH,
                          model_2_path  = MODEL_2_PATH,
                          model_3_path  = MODEL_3_PATH,
                          num_epochs    = NUM_EPOCHS,
                          transforms    = TRANSFORMS,
                          batch_size    = BATCH_SIZE,
                          lr            = LR,
                          num_workers   = NUM_WORKERS,
                          dropout_rate  = DROPOUT_RATE)

    # RuntimeError: Modules with uninitialized parameters can't be used with `DistributedDataParallel`. Run a dummy forward pass to correctly initialize the modules
    dum = np.random.randint(255, size=[16, 3, 1024, 1024])/255
    dum = torch.from_numpy(dum).float()
    _ = model(dum)

    trainer.fit(model)

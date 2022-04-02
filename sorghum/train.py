# %%
import csv
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import socket

import pytorch_lightning as pl # Works with pl.__version__ == '1.5.10'
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from src.data import SorghumDataset
from src.constants import CULTIVAR_LABELS_IND2STR, CULTIVAR_LABELS_STR2IND, IMAGENET_NORMAL_MEAN, IMAGENET_NORMAL_STD, BACKBONE_IMG_SIZE

# %%
class SorghumLitModel(pl.LightningModule):
    def __init__(self, backbone, transforms, num_classes, batch_size, lr, n_hidden_nodes, 
                 dropout_rate=0, pretrained=True, num_workers=4):
        super(SorghumLitModel, self).__init__()
        self.save_hyperparameters() # Need this later to load_from_checkpoint without providing the hyperparams again

        self.transforms = transforms # Define transforms using albumentations here!
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers= num_workers

        self.backbone = backbone
        self.n_hidden_nodes = n_hidden_nodes
        self.dropout = nn.Dropout(p=dropout_rate)

        self.now = datetime.now().strftime('%Y%m%d_%H%M%S')
        print('Run ID: ', self.now)
        self.tests_result_csv_filename = 'test_result_{}.csv'.format(self.now)
        self.csv_header_written = False

        if self.backbone == 'xception': # INPUT_SIZE = 3 x 299 x 299
            from pretrainedmodels import xception # densenet121, densenet201
            self.model = xception(num_classes=1000, pretrained='imagenet' if pretrained else False)
            self.model.last_linear = nn.Identity() # Outputs 2048
            n_backbone_out = 2048
            self.target_size=299

        if self.n_hidden_nodes is None:
            self.img_fc1 = nn.Linear(n_backbone_out, num_classes)
        else:
            self.img_fc1 = nn.Linear(n_backbone_out, n_hidden_nodes)
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(n_hidden_nodes, num_classes)

        trainval_dataset = SorghumDataset(csv_fullpath='../../dataset/sorghum/train_cultivar_mapping.csv',
                                   transform=self.transforms,
                                   testset=False)

        self.train_dataset, self.val_dataset = \
            random_split(trainval_dataset, [round(len(trainval_dataset)*0.8), round(len(trainval_dataset)*0.2)])

    def forward(self, x):
        if self.n_hidden_nodes is not None:
            out = self.model(x)
            out = self.dropout(out)
            out = self.img_fc1(out)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.fc1(out) # No activation and no softmax at the end
        else:
            out = self.model(x)
            out = self.dropout(out)
            out = self.img_fc1(out)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
        # Can also set up the lr_scheduler here as well, like below:
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        # return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        train_loader = DataLoader(dataset       = self.train_dataset,
                                  batch_size    = self.batch_size, 
                                  shuffle       = True, 
                                  num_workers   = self.num_workers,
                                  persistent_workers = True)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(dataset         = self.val_dataset, 
                                batch_size      = self.batch_size, 
                                shuffle         = False, 
                                num_workers     = self.num_workers,
                                persistent_workers = True)
        return val_loader

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

# %% Hyperparameters
PRETRAINED = True
N_HIDDEN_NODES = 500 # No hidden layer if None
DROPOUT_RATE = 0.5 # No dropout if 0
NUM_CLASSES = 100 # Fixed (for this challenge)
NUM_EPOCHS = 30
LR = 0.001 # Set up to be automatically adjusted (see Trainer parameter)
NUM_WORKERS= 16 # use os.cpu_count()
BACKBONE = 'xception'

host_name = socket.gethostname()
if BACKBONE == 'xception':
    BATCH_SIZE = 64 if host_name=='jupyter-brian' else 256 # effective batch size = batch_size * gpus * num_nodes. 256 on A100, 64 on GTX 1080Ti

TRANSFORMS = A.Compose([
                A.RandomResizedCrop(height=BACKBONE_IMG_SIZE[BACKBONE], width=BACKBONE_IMG_SIZE[BACKBONE]), # Improved final score by 0.023 (0.575->0.598)
                # A.Resize(height=BACKBONE_IMG_SIZE[BACKBONE], width=BACKBONE_IMG_SIZE[BACKBONE]),
                A.HorizontalFlip(p=0.5), # Leaving this on improved performance (at 0.5)
                A.VerticalFlip(p=0.5), # Leaving this on improved performance (at 0.5)
                # A.ColorJitter (brightness=0.2, contrast=0.2, p=0.3),
                # A.ChannelShuffle(p=0.3),
                # A.Normalize(IMAGENET_NORMAL_MEAN, IMAGENET_NORMAL_STD), # Turning this on obliterated performance (only for validation metrics)
                ToTensorV2(), # np.array HWC image -> torch.Tensor CHW
            ]) # Try one where the normalization happens before colorjitter and channelshuffle -> not a good idea
TB_NOTES = 'Dropout, RandomResizedCrop, Flips, UnNormalized'

# %%
if __name__=='__main__':
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    if TB_NOTES != '':
        now += '_' + TB_NOTES

    logger_tb = TensorBoardLogger('./tb_logs', name=now)
    logger_wandb = WandbLogger(project='Sorghum', name=now, mode='online') # online or disabled

    # Saves checkpoints at every epoch
    checkpoint_callback = ModelCheckpoint(dirpath='./tb_logs/{}/'.format(now), 
                                          monitor='val_loss', 
                                          filename='{epoch:02d}-{val_loss:.2f}',
                                          save_top_k=3)

    trainer = Trainer(max_epochs            = NUM_EPOCHS, 
                      fast_dev_run          = False,     # Run a single-batch through train and val and see if the code works.
                      gpus                  = -1,        # -1 to use all available GPUs, [0, 1, 2] to specify GPUs by index
                      auto_lr_find          = True,
                      default_root_dir      = '../', 
                      precision             = 16,  # mixed precision training
                      logger                = [logger_tb, logger_wandb],
                      log_every_n_steps     = 10,
                      accelerator           = 'ddp',
                      callbacks             = [checkpoint_callback],
                      plugins               = DDPPlugin(find_unused_parameters=False))

    model = SorghumLitModel(backbone        = BACKBONE, 
                            transforms      = TRANSFORMS, 
                            num_classes     = NUM_CLASSES,
                            batch_size      = BATCH_SIZE, 
                            lr              = LR, 
                            pretrained      = PRETRAINED, 
                            n_hidden_nodes  = N_HIDDEN_NODES, 
                            dropout_rate    = DROPOUT_RATE,
                            num_workers     = NUM_WORKERS)

    trainer.fit(model)

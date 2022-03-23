# %%
import sys
import csv
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import pytorch_lightning as pl # Works with plt.__version__ == '1.5.10'
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

sys.path.append('./src/')
from data import SorghumDataset
from constants import CULTIVAR_LABELS, CULTIVAR_LABELS_ALT

from pretrainedmodels import xception, densenet121, densenet201


# %% Hyperparameters
INPUT_SIZE = 299 # For xception
N_HIDDEN_NODES = 500 # If none, no hidden layer
NUM_CLASSES = 100
NUM_EPOCHS = 30
BATCH_SIZE = 256 # effective batch size = batch_size * gpus * num_nodes
LR = 0.001
NUM_WORKERS= 16 # use os.cpu_count()
TRANSFORMS = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ColorJitter (brightness=0.2, contrast=0.2, p=0.3),
                A.ChannelShuffle(p=0.3),
                # A.Normalize(mean = [0.485, 0.456, 0.406],
                #             std =  [0.229, 0.224, 0.225]), # Imagenet standard
            ]) # Try one where the normalization happens before colorjitter and channelshuffle

# %%
class SorghumLitModel(pl.LightningModule):
    def __init__(self, backbone, input_size, transforms, num_classes, batch_size, lr, n_hidden_nodes, 
                 pretrained=True, num_workers=4):
        super(SorghumLitModel, self).__init__()
        self.save_hyperparameters() # Need this later to load_from_checkpoint without providing the hyperparams again

        self.transforms = transforms # Define transforms using albumentations here!
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers= num_workers

        self.input_size = input_size
        self.backbone = backbone
        self.n_hidden_nodes = n_hidden_nodes

        self.now = datetime.now().strftime('%Y%m%d_%H%M%S')
        print('Run ID: ', self.now)
        self.tests_result_csv_filename = 'test_result_{}.csv'.format(self.now)
        self.csv_header_written = False

        if self.backbone == 'xception': # INPUT_SIZE = 3 x 299 x 299
            self.model = xception(num_classes=1000, pretrained='imagenet' if pretrained else False)
            self.model.last_linear = nn.Identity() # Outputs 2048
            n_backbone_out = 2048
        if self.n_hidden_nodes is None:
            self.img_fc1 = nn.Linear(n_backbone_out, num_classes)
        else:
            self.img_fc1 = nn.Linear(n_backbone_out, n_hidden_nodes)
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(n_hidden_nodes, num_classes)

        trainval_dataset = SorghumDataset(csv_fullpath='/home/brian/dataset/sorghum/train_cultivar_mapping.csv',
                                   transform=self.transforms,
                                   target_size=self.input_size,
                                   testset=False)

        self.train_dataset, self.val_dataset = \
            random_split(trainval_dataset, [round(len(trainval_dataset)*0.8), round(len(trainval_dataset)*0.2)])

    def forward(self, x):
        if self.n_hidden_nodes is not None:
            out = self.model(x)
            out = self.img_fc1(out)
            out = self.relu(out)
            out = self.fc1(out) # No activation and no softmax at the end
        else:
            out = self.model(x)
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
                writer.writerow([filename, CULTIVAR_LABELS[classification]])

# %%
if __name__=='__main__':
    now = datetime.now().strftime('%Y%m%d_%H%M%S')

    # fast_dev_run=True will run a single-batch through training and validation and test if the code works.
    logger = TensorBoardLogger('./tb_logs', name=now)

    # Saves checkpoints at every epoch
    checkpoint_callback = ModelCheckpoint(dirpath='./tb_logs/{}/'.format(now), 
                                          monitor='val_loss', 
                                          filename='{epoch:02d}-{val_loss:.2f}',
                                          save_top_k = 4)

    trainer = Trainer(max_epochs            = NUM_EPOCHS, 
                      fast_dev_run          = False, 
                      gpus                  = -1, 
                      auto_lr_find          = True,
                      default_root_dir      = '../', 
                      precision             = 16,  # mixed precision training
                      logger                = logger,
                      log_every_n_steps     = 10,
                      accelerator           = 'ddp',
                      callbacks             = [checkpoint_callback],
                      plugins               = DDPPlugin(find_unused_parameters=False))

    model = SorghumLitModel(backbone        = 'xception', 
                            input_size      = INPUT_SIZE, 
                            transforms      = TRANSFORMS, 
                            num_classes     = NUM_CLASSES,
                            batch_size      = BATCH_SIZE, 
                            lr              = LR, 
                            pretrained      = True, 
                            n_hidden_nodes  = N_HIDDEN_NODES, 
                            num_workers     = NUM_WORKERS)

    trainer.fit(model)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning import Trainer

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from data import HerbariumDataset

from pretrainedmodels import xception, densenet121, densenet201
# from torchsummary import summary

# %% Hyperparameters
INPUT_SIZE = 299 # For xception
N_HIDDEN_NODES = None # If none, no hidden layer
NUM_CLASSES = 15505
NUM_EPOCHS = 5
BATCH_SIZE = 32
LR = 0.001
NUM_WORKERS= 4
TRANSFORMS = None

# %%
class LightningHerb(pl.LightningModule):
    def __init__(self, backbone, input_size, transforms, num_classes, batch_size, lr, n_hidden_nodes, 
                 pretrained=True, num_workers=4):
        super(LightningHerb, self).__init__()

        # transforms = A.Compose([
        #     A.HorizontalFlip(p=0.5),
        #     A.VerticalFlip(p=0.5),
        #     A.OneOf([
        #             A.RandomRotate90(p=0.5), 
        #             A.Rotate(p=0.5)],
        #         p=0.5),
        #     A.ColorJitter (brightness=0.2, contrast=0.2, p=0.3),
        #     A.ChannelShuffle(p=0.3),
        #     # A.Normalize(NORMAL_MEAN, NORMAL_STD),
        #     ToTensorV2()
        # ])

        self.transforms = transforms # Define transforms using albumentations here!
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers= num_workers

        self.input_size = input_size
        self.backbone = backbone
        self.n_hidden_nodes = n_hidden_nodes

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

        trainval_dataset = HerbariumDataset(csv_fullpath='/home/jovyan/brian/kaggle_herbarium_2022/train.csv', 
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
        images, category = batch
        outputs = self(images) # Forward pass
        loss = F.cross_entropy(outputs, category)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
        
    def validation_step(self, batch, batch_idx):
        images, category = batch
        outputs = self(images) # Forward pass
        loss = F.cross_entropy(outputs, category)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs): # Run this at the end of a validation epoch
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

# %%
if __name__=='__main__':
    # fast_dev_run=True will run a single-batch through training and validation and test if the code works.
    trainer = Trainer(max_epochs=NUM_EPOCHS, fast_dev_run=False, gpus=4, auto_lr_find=True)

    model = LightningHerb(backbone='xception', input_size=INPUT_SIZE, transforms=TRANSFORMS, num_classes=NUM_CLASSES,
                          batch_size=BATCH_SIZE, lr=LR, pretrained=True, n_hidden_nodes=N_HIDDEN_NODES, num_workers=NUM_WORKERS)

    trainer.fit(model)

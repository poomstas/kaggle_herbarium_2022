# %%
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from train import SorghumLitModel

sys.path.append('./src/')
from data import SorghumDataset

# %%
# Tried
# CHK_PATH = '/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220323_105943/epoch=18-val_loss=0.31.ckpt'
# CHK_PATH = '/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220323_225103/epoch=16-val_loss=0.20.ckpt' # 0.507
# CHK_PATH = '/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220324_094248/epoch=16-val_loss=0.11.ckpt' # 0.553
# CHK_PATH = '/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220324_111631_Added dropout layer, turned off horz vert flips/epoch=23-val_loss=0.09.ckpt'
# CHK_PATH = '/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220325_214512_Added dropout layer, turned on normalization, left on flips/epoch=23-val_loss=0.07.ckpt'

# To Try
CHK_PATH = ''
# CHK_PATH = ''
# CHK_PATH = ''
# CHK_PATH = ''
# CHK_PATH = ''

# %%
# Need to have caled self.save_hyperparameters() in model init for the below to work!
model = SorghumLitModel.load_from_checkpoint(checkpoint_path=CHK_PATH) 

test_dataset = SorghumDataset(csv_fullpath='test.csv', testset=True)
dl_test = DataLoader(dataset=test_dataset, shuffle=False, batch_size=512, num_workers=16)

trainer = Trainer(gpus=1)
results = trainer.test(model=model, dataloaders=dl_test, verbose=True)

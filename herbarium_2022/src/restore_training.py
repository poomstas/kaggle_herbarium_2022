'''
Resume training. Script incomplete...
'''
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

from train import LightningHerb

# %%
model = LightningHerb()
trainer = Trainer()

CHECKPOINT_PATH = '/home/jovyan/brian/kaggle_herbarium_2022/src/lightning_logs/version_1/checkpoints/epoch=1-step=10497.ckpt'

# automatically restores model, epoch, step, LR schedulers, apex, etc...
trainer.fit(model, ckpt_path=CHECKPOINT_PATH)

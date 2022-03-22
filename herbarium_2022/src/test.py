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
CHECKPOINT_PATH = '/home/jovyan/brian/kaggle_herbarium_2022/src/lightning_logs/version_1/checkpoints/epoch=1-step=10497.ckpt'

model = LightningHerb.load_from_checkpoint(checkpoint_path=CHECKPOINT_PATH)

# model = LightningHerb(backbone='xception', input_size=INPUT_SIZE, transforms=TRANSFORMS, num_classes=NUM_CLASSES,
#                         batch_size=BATCH_SIZE, lr=LR, pretrained=True, n_hidden_nodes=N_HIDDEN_NODES, num_workers=NUM_WORKERS)
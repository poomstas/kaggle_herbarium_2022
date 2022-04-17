# %%
MODEL_1_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220414_140928_InputRes1024_3FC_ReducedMaxLR_BaseCase_nipa2022-49703_efficientnet-b3_2048_UnfreezeAt99999_ResizerApplied_False/epoch=46-val_loss=0.00378.ckpt"
MODEL_2_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220415_104735_InputRes1024_3FC_ReducedMaxLR_BaseCase_nipa2022-49703_efficientnet-b7_2048_UnfreezeAt99999_ResizerApplied_False/epoch=40-val_loss=0.00184.ckpt"

# %%
# import os
# import csv
# import socket
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Subset
# from datetime import datetime

# import pytorch_lightning as pl
# from pytorch_lightning import Trainer
# from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
# from pytorch_lightning.plugins import DDPPlugin

# import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2

# from torchsummary import summary

# from src.data import SorghumDataset
# from src.constants import CULTIVAR_LABELS_IND2STR, IMAGENET_NORMAL_MEAN, IMAGENET_NORMAL_STD, BACKBONE_IMG_SIZE
# from src.utils import balance_val_split, get_stratified_sampler_for_subset
# from src.learnable_resizer import LearnToResize, LearnToResizeKaggle

from train import SorghumLitModel
# %%
MODEL_1 = SorghumLitModel.load_from_checkpoint(checkpoint_path=MODEL_1_PATH) 
MODEL_2 = SorghumLitModel.load_from_checkpoint(checkpoint_path=MODEL_2_PATH) 

''' MODEL_1
    (img_fc1): Linear(in_features=1000, out_features=1024, bias=True)
    (relu1): ReLU()
    (fc2): Linear(in_features=1024, out_features=512, bias=True)
    (relu2): ReLU()
    (fc3): Linear(in_features=512, out_features=100, bias=True)
    (relu3): ReLU()
'''

''' MODEL_2
    (img_fc1): Linear(in_features=1000, out_features=1024, bias=True)
    (relu1): ReLU()
    (fc2): Linear(in_features=1024, out_features=512, bias=True)
    (relu2): ReLU()
    (fc3): Linear(in_features=512, out_features=100, bias=True)
    (relu3): ReLU()
'''
# %%

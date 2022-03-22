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
CHK_PATH = '/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/version_1/checkpoints/epoch=2-step=104 copy.ckpt'

# %%
# Need to have caled self.save_hyperparameters() in model init for the below to work!
model = SorghumLitModel.load_from_checkpoint(checkpoint_path=CHK_PATH) 

print('='*90)

test_dataset = SorghumDataset(csv_fullpath='test.csv', testset=True)
dl_test = DataLoader(dataset=test_dataset, shuffle=False, batch_size=128, num_workers=16)

trainer = Trainer(gpus=1)
results = trainer.test(model=model, test_dataloaders=dl_test, verbose=True)

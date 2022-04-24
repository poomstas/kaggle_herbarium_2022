# %%
MODEL_1_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220414_140928_InputRes1024_3FC_ReducedMaxLR_BaseCase_nipa2022-49703_efficientnet-b3_2048_UnfreezeAt99999_ResizerApplied_False/epoch=48-val_loss=0.00391.ckpt" # 0.858
MODEL_2_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220420_035255_InputRes1024_3FC_ReducedMaxLR_nipa2022-49703_efficientnet-b3_2048_UnfreezeAt99999_ResizerApplied_False_DropoutRate0.5_BaseCase_20220414_140928/epoch=53-val_loss=0.00167.ckpt" # 0.840
MODEL_3_PATH = ""

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

from src.data import SorghumDataset
from src.constants import CULTIVAR_LABELS_IND2STR, IMAGENET_NORMAL_MEAN, IMAGENET_NORMAL_STD, BACKBONE_IMG_SIZE
from src.utils import balance_val_split, get_stratified_sampler_for_subset
from src.learnable_resizer import LearnToResize, LearnToResizeKaggle

from train import SorghumLitModel
# %%
MODEL_1 = SorghumLitModel.load_from_checkpoint(checkpoint_path=MODEL_1_PATH) # Input Res: 2048 x 2048
MODEL_2 = SorghumLitModel.load_from_checkpoint(checkpoint_path=MODEL_2_PATH) # Input Res: 2048 x 2048
MODEL_3 = SorghumLitModel.load_from_checkpoint(checkpoint_path=MODEL_3_PATH) # Input Res: 2048 x 2048

''' Below needs to be re-verified:

MODEL_1
    (img_fc1): Linear(in_features=1000, out_features=1024, bias=True)
    (relu1): ReLU()
    (fc2): Linear(in_features=1024, out_features=512, bias=True)
    (relu2): ReLU()
    (fc3): Linear(in_features=512, out_features=100, bias=True)
    (relu3): ReLU()

MODEL_2
    (img_fc1): Linear(in_features=1000, out_features=1024, bias=True)
    (relu1): ReLU()
    (fc2): Linear(in_features=1024, out_features=512, bias=True)
    (relu2): ReLU()
    (fc3): Linear(in_features=512, out_features=100, bias=True)
    (relu3): ReLU()

MODEL_3
    ???
    ???
    ???
'''
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
NUM_WORKERS         = os.cpu_count()
BATCH_SIZE          = 32
# FREEZE_BACKBONE     = False
# UNFREEZE_AT         = 99999         # Disables freezing if 0 (epoch count starts at 0)

host_name = socket.gethostname()
print('\nHost Name: {}\n'.format(host_name))

now = datetime.now().strftime('%Y%m%d_%H%M%S')
TB_NOTES = "Ensemble_{}".format(now)
TB_NOTES += "_" + host_name
TB_NOTES += "_DropoutRate{}".format(str(DROPOUT_RATE))

# %%
class EnsembleModel(pl.LightningModule):
    def __init__(self, model_1_path, model_2_path, model_3_path, num_epochs, transforms, batch_size, lr, num_workers):
        super(EnsembleModel, self).__init__()
        self.save_hyperparameters() # Need this later to load_from_checkpoint without providing the hyperparams again
        
        self.model_1         = SorghumLitModel.load_from_checkpoint(checkpoint_path=model_1_path)
        self.model_2         = SorghumLitModel.load_from_checkpoint(checkpoint_path=model_2_path)
        self.model_3         = SorghumLitModel.load_from_checkpoint(checkpoint_path=model_3_path)

        self.num_epochs      = num_epochs
        self.transforms      = transforms
        self.batch_size      = batch_size
        self.lr              = lr
        self.num_workers     = num_workers

        self.fc1 = nn.Linear()

    def forward(self, x):
        pass

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



# %%

# %%
import sys
sys.path.insert(0, '..')

import torch
from train import SorghumLitModel

# %%
CHECKPOINT_FILE_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220407_081121_3OneOf_OneCycleLR_2FCLayer1stLayer4096_BaseCase20220404_075755_hades-ubuntu/epoch=00-val_loss=4.13.ckpt"
OUTPUT_FILE = "./pth_model.pth"

model = SorghumLitModel.load_from_checkpoint(checkpoint_path=CHECKPOINT_FILE_PATH) 

torch.save(model, OUTPUT_FILE)
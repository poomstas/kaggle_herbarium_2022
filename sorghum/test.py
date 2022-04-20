# %%
import os
import time
import albumentations as A
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from albumentations.pytorch.transforms import ToTensorV2
from train import SorghumLitModel
from src.data import SorghumDataset
from src.constants import BACKBONE_IMG_SIZE, IMAGENET_NORMAL_MEAN, IMAGENET_NORMAL_STD

# %%
start_time = time.time()
# Tried
# CHK_PATH = '/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220323_105943/epoch=18-val_loss=0.31.ckpt'
# CHK_PATH = '/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220323_225103/epoch=16-val_loss=0.20.ckpt' # 0.507
# CHK_PATH = '/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220324_094248/epoch=16-val_loss=0.11.ckpt' # 0.553
# CHK_PATH = '/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220324_111631_Added dropout layer, turned off horz vert flips/epoch=23-val_loss=0.09.ckpt'
# CHK_PATH = '/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220325_214512_Added dropout layer, turned on normalization, left on flips/epoch=23-val_loss=0.07.ckpt'
# CHK_PATH = '/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220329_024615_Added dropout layer, turned on normalization, left on flips/epoch=25-val_loss=0.08.ckpt' # 0.575
# CHK_PATH = '/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220402_103938_Dropout, RandomResizedCrop, Flips, UnNormalized/epoch=28-val_loss=0.25.ckpt'
# CHK_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220402_135655_SH1R0's transforms, without normalization and ISONoise/epoch=28-val_loss=2.27.ckpt"
# CHK_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220402_195622_SH1R0's transforms, without normalization and ISONoise/epoch=23-val_loss=0.38.ckpt"
# CHK_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220403_113842_DDP_StratifiedSampler_SH1R0s transforms, without normalization and ISONoise/epoch=18-val_loss=0.28.ckpt"
# CHK_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220403_183214_First3Trans_DDPStratifiedSampler/epoch=36-val_loss=0.17.ckpt"
# CHK_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220403_205250_3FCLayer1stLayer4096/epoch=37-val_loss=0.26.ckpt"
# CHK_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220403_232455_OneCycleLR_2FCLayer1stLayer4096/epoch=25-val_loss=0.08.ckpt"
# CHK_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220404_003524_OneCycleLR_2FCLayer1stLayer4096/epoch=29-val_loss=0.06.ckpt"
# CHK_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220404_021721_OneCycleLR_2FCLayer1stLayer4096/epoch=59-val_loss=0.02.ckpt" # 0.725
# CHK_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220404_075755_OneCycleLR_2FCLayer1stLayer4096/epoch=56-val_loss=0.02.ckpt"
# CHK_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220404_103901_1OneOf_OneCycleLR_2FCLayer1stLayer4096_BaseCase20220404_075755/epoch=56-val_loss=0.02.ckpt"
# CHK_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220404_134736_2OneOf_OneCycleLR_2FCLayer1stLayer4096_BaseCase20220404_075755/epoch=59-val_loss=0.02.ckpt"
# CHK_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220404_171151_2OneOf_OneCycleLR_2FCLayer1stLayer4096_BaseCase20220404_075755/epoch=58-val_loss=0.02.ckpt"
# CHK_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220411_063830_3OneOF_OneCycleLR_3FC_BaseCaseSelf_nipa2022-49703_xception_2048_UnfreezeAt0/epoch=36-val_loss=0.06.ckpt"
# CHK_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220411_081219_3OneOF_OneCycleLR_3FC_BaseCaseSelf_nipa2022-49703_xception_2048_UnfreezeAt0/epoch=53-val_loss=0.01.ckpt"
# CHK_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220411_114332_EffNetB3_3OneOF_OneCycleLR_3FC_BaseCase_nipa2022-49703_efficientnet-b3_2048_UnfreezeAt99999/epoch=55-val_loss=0.00.ckpt" # 350x350 test_result_20220411_155751.csv // 0.771
# CHK_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220411_165911_3OneOF_OneCycleLR_3FC_BaseCase_nipa2022-49703_efficientnet-b3_2048_UnfreezeAt99999/epoch=52-val_loss=0.00.ckpt" # 512 x 512 test_result_20220411_221241.csv //
# CHK_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220411_165911_3OneOF_OneCycleLR_3FC_BaseCase_nipa2022-49703_efficientnet-b3_2048_UnfreezeAt99999/epoch=57-val_loss=0.00.ckpt" # 512 x 512 test_result_20220411_221457.csv //
# CHK_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220411_165911_3OneOF_OneCycleLR_3FC_BaseCase_nipa2022-49703_efficientnet-b3_2048_UnfreezeAt99999/epoch=59-val_loss=0.00.ckpt" # 512 x 512 test_result_20220411_221709.csv // 0.843
# CHK_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220411_224324_3OneOF_OneCycleLR_3FC_BaseCase_nipa2022-49703_resnest-269_2048_UnfreezeAt99999/epoch=59-val_loss=0.00255.ckpt"
# CHK_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220412_062407_3OneOF_OneCycleLR_3FC_ChangedLRScheme_BaseCase_nipa2022-49703_resnest-269_2048_UnfreezeAt99999/epoch=50-val_loss=0.00271.ckpt" # 0.830
# CHK_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220412_150637_3OneOF_OneCycleLR_3FC_ChangedLRScheme_BaseCase_nipa2022-49703_efficientnet-b3_2048_UnfreezeAt99999/epoch=50-val_loss=0.00402.ckpt"
# CHK_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220413_103158_3OneOF_OneCycleLR_3FC_ChangedLRScheme_BaseCase_nipa2022-49703_efficientnet-b3_2048_UnfreezeAt99999_ResizerApplied_True/epoch=52-val_loss=0.00311.ckpt"
# CHK_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220413_225929_3OneOF_OneCycleLR_3FC_ChangedLRScheme_BaseCase_nipa2022-49703_resnest-269_2048_UnfreezeAt99999_ResizerApplied_True/epoch=31-val_loss=0.01208.ckpt"
# CHK_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220414_140928_InputRes1024_3FC_ReducedMaxLR_BaseCase_nipa2022-49703_efficientnet-b3_2048_UnfreezeAt99999_ResizerApplied_False/epoch=46-val_loss=0.00378.ckpt" # 0.856 Top
# CHK_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220414_140928_InputRes1024_3FC_ReducedMaxLR_BaseCase_nipa2022-49703_efficientnet-b3_2048_UnfreezeAt99999_ResizerApplied_False/epoch=48-val_loss=0.00391.ckpt"
# CHK_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220415_104735_InputRes1024_3FC_ReducedMaxLR_BaseCase_nipa2022-49703_efficientnet-b7_2048_UnfreezeAt99999_ResizerApplied_False/epoch=40-val_loss=0.00184.ckpt" # 0.854 1024 x 1024 test_result_20220417_171214.csv
# CHK_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220415_104735_InputRes1024_3FC_ReducedMaxLR_BaseCase_nipa2022-49703_efficientnet-b7_2048_UnfreezeAt99999_ResizerApplied_False/epoch=38-val_loss=0.00212.ckpt" # 0.845 1024x1024  test_result_20220417_203941.csv 
# CHK_PATH = "/home/brian/sorghum/tb_logs/20220419_080226_InputRes1024_3FC_ReducedMaxLR_nipa2022-49703_efficientnet-b3_2048_UnfreezeAt99999_ResizerApplied_False_KaggleResizer_DropoutRate0.5/epoch=45-val_loss=0.00783.ckpt" 

# To Try
CHK_PATH = ""
# CHK_PATH = ""
# CHK_PATH = ""

# %%
# Need to have caled self.save_hyperparameters() in model init for the below to work!
csv_file_name = CHK_PATH.split('/')[6] + ".csv"

model = SorghumLitModel.load_from_checkpoint(checkpoint_path=CHK_PATH) 
# BACKBONE = 'efficientnet-b3'

TRANSFORM = A.Compose([
                # A.Resize(height=BACKBONE_IMG_SIZE[BACKBONE], width=BACKBONE_IMG_SIZE[BACKBONE]),
                A.Normalize(IMAGENET_NORMAL_MEAN, IMAGENET_NORMAL_STD),
                ToTensorV2(), # np.array HWC image -> torch.Tensor CHW
            ])

test_dataset = SorghumDataset(csv_fullpath='test.csv', testset=True, transform=TRANSFORM) # xception: 299
dl_test = DataLoader(dataset=test_dataset, shuffle=False, batch_size=32, num_workers=os.cpu_count())

trainer = Trainer(gpus=1)
results = trainer.test(model=model, dataloaders=dl_test, verbose=True)

run_time = (time.time() - start_time)
print('Execution time in seconds: ' + str(run_time))

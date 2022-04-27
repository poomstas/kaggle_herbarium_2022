# %%
import os
import time
import albumentations as A
from datetime import datetime
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from albumentations.pytorch.transforms import ToTensorV2
from src.data import SorghumDataset
from src.constants import IMAGENET_NORMAL_MEAN, IMAGENET_NORMAL_STD
from ensemble_train import EnsembleModel

# %%
TRANSFORM  = A.Compose([
        # A.Resize(height=BACKBONE_IMG_SIZE[BACKBONE], width=BACKBONE_IMG_SIZE[BACKBONE]),
        A.Normalize(IMAGENET_NORMAL_MEAN, IMAGENET_NORMAL_STD),
        ToTensorV2(), # np.array HWC image -> torch.Tensor CHW]
])

# %%
ENSEMBLE_MODEL_PATH = "/home/brian/github/kaggle_herbarium_2022/sorghum/tb_logs/20220427_060458_Ensemble_20220427_060458_nipa2022-49703_DropoutRate0.5/epoch=49-val_loss=0.00172.ckpt"

model = EnsembleModel.load_from_checkpoint(checkpoint_path=ENSEMBLE_MODEL_PATH)

# Remove below for later runs
# model.csv_header_written = False
# model.csv_header_written = False
# now = datetime.now().strftime('%Y%m%d_%H%M%S')
# model.tests_result_csv_filename = 'test_result_{}.csv'.format(now)

# %%
start_time = time.time()

test_dataset = SorghumDataset(csv_fullpath='test.csv', testset=True, transform=TRANSFORM)
dl_test = DataLoader(dataset=test_dataset, shuffle=False, batch_size=42, num_workers=os.cpu_count())

trainer = Trainer(gpus=1)
results = trainer.test(model=model, dataloaders=dl_test, verbose=True)

run_time = (time.time() - start_time)
print('Execution time in seconds: ' + str(run_time))

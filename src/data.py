# %%
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# Load the file even if the image file is truncated. See: https://discuss.pytorch.org/t/oserror-image-file-is-truncated-150-bytes-not-processed/64445
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# %%
class HerbariumDataset(Dataset):
    def __init__(self, csv_fullpath, transform=None, target_size=299, testset=False):
        self.df                         = pd.read_csv(csv_fullpath)
        self.transform                  = transform
        self.target_size                = target_size
        self.testset                    = testset # Boolean
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_fullpath = self.df['directory'][index]
        img = mpimg.imread(img_fullpath) # Reads in [H, W, C]

        if (self.target_size, self.target_size) != img.shape[:2]:
            img = cv2.resize(img, (self.target_size, self.target_size))

        if self.transform is not None:
            img = self.transform(image=img)["image"]
        
        img = np.transpose(img, (2, 1, 0)) # Convert from [H, W, C] to [C, W, H] and convert it to float
        # TODO: Reduce computation by loading in the image in a format that doesn't require transposing
        img = torch.from_numpy(img).float() # Convert from np.array to torch float
        
        if self.testset:
            return img
        else:
            category = self.df['category'][index]
            return img, category

# %%
if __name__=='__main__':
    ''' Test to see if the Dataset and DataLoader objects are working correctly. '''
    NORMAL_MEAN = [0.5, 0.5, 0.5]
    NORMAL_STD = [0.5, 0.5, 0.5]
    
    transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
                A.RandomRotate90(p=0.5), 
                A.Rotate(p=0.5)],
            p=0.5),
        A.ColorJitter (brightness=0.2, contrast=0.2, p=0.3),
        A.ChannelShuffle(p=0.3),
        A.Normalize(NORMAL_MEAN, NORMAL_STD),
        ToTensorV2()
    ])

    ds_train = HerbariumDataset('../train.csv', transform=transforms)
    dl_train = DataLoader(dataset=ds_train,
                          shuffle=True,
                          batch_size=3,
                          num_workers=4)

    for i, (img, category) in enumerate(dl_train):
        if i > 10:
            break
        print(img.shape)
        print(category)

    # How to do splits correctly, for reference
    train_dataset, val_dataset = random_split(ds_train, [round(len(ds_train)*0.8), round(len(ds_train)*0.2)])

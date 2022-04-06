# %%
import os
import pandas as pd
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from src.constants import CULTIVAR_LABELS_IND2STR, CULTIVAR_LABELS_STR2IND, IMAGENET_NORMAL_MEAN, IMAGENET_NORMAL_STD

# Load the file even if the image file is truncated. 
# See: https://discuss.pytorch.org/t/oserror-image-file-is-truncated-150-bytes-not-processed/64445
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# %%
class SorghumDataset(Dataset):
    def __init__(self, csv_fullpath, dataset_root='./data/sorghum/',
                 transform=None, testset=False):
        self.transform                  = transform
        self.testset                    = testset # Boolean

        if testset:
            dataset_root = os.path.join(dataset_root, 'test')
            file_list = os.listdir(dataset_root)
            file_fullpaths = [os.path.join(dataset_root, filename) for filename in file_list]
            self.df = pd.DataFrame(file_fullpaths, columns=['image'])
        else:
            self.df = pd.read_csv(csv_fullpath)
            self.df['cultivar_indx'] = [CULTIVAR_LABELS_STR2IND[cultivar_name] for cultivar_name in self.df['cultivar']] # e.g.) 'PI_329319' to 91
            dataset_root = os.path.join(dataset_root, 'train_images')
            self.df['image'] = [os.path.join(dataset_root, img_path) for img_path in self.df['image']]

            # Check if dataset exists. If not, then remove it from the dataframe.
            print('Original dataset length (CSV):', len(self.df))
            image_unavailable_indx = []
            for indx, row in self.df.iterrows(): 
                if not os.path.exists(row['image']):
                    image_unavailable_indx.append(indx)
            self.df.drop(image_unavailable_indx, axis=0, inplace=True)
            self.df = self.df.reset_index()
            print('Validated dataset length (CSV):', len(self.df))
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_fullpath = self.df['image'][index]
        img = mpimg.imread(img_fullpath) # Reads in [H, W, C]

        if self.transform is not None:
            img = self.transform(image=img)["image"]
        
        if self.testset:
            filename = self.df['image'][index].split('/')[-1]
            return img, filename
        else:
            cultivar_indx = self.df['cultivar_indx'][index]
            return img, cultivar_indx

# %%
if __name__=='__main__':
    ''' Test to see if the Dataset and DataLoader objects are working correctly. '''

    transforms = A.Compose([
        A.Resize(height=299, width=299),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
                A.RandomRotate90(p=0.5), 
                A.Rotate(p=0.5)],
            p=0.5),
        A.ColorJitter (brightness=0.2, contrast=0.2, p=0.3),
        A.ChannelShuffle(p=0.3),
        A.Normalize(IMAGENET_NORMAL_MEAN, IMAGENET_NORMAL_STD),
        ToTensorV2(), # np.array HWC image -> torch.Tensor CHW
    ])

    # Train Dataset
    print('Testing Training Dataset')
    ds_train = SorghumDataset(csv_fullpath='./data/sorghum/train_cultivar_mapping.csv', 
                              transform=transforms)
    dl_train = DataLoader(dataset=ds_train,
                          shuffle=True,
                          batch_size=3,
                          num_workers=4)

    for i, (img, cultivar_indx) in enumerate(dl_train):
        if i > 10:
            break
        print(img.shape)
        print(cultivar_indx)

    # Test Dataset
    print('='*90)
    print('Testing Test Dataset')
    ds_train = SorghumDataset(csv_fullpath='../test.csv',
                              transform=transforms,
                              testset=True)
    dl_train = DataLoader(dataset=ds_train,
                          shuffle=True,
                          batch_size=3,
                          num_workers=4)

    for i, (img, filename) in enumerate(dl_train):
        if i > 10:
            break
        print(img.shape)
        print(filename)

    # How to do splits correctly, for reference
    train_dataset, val_dataset = random_split(ds_train, [round(len(ds_train)*0.8), round(len(ds_train)*0.2)])

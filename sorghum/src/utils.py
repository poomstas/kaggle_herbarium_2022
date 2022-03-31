# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from data import SorghumDataset
from torch.utils.data import DataLoader
from constants import CULTIVAR_LABELS_STR2IND
from torch.utils.data import Subset, DataLoader

# %%
# def stratified_split_train_val(df, target_variable_name, test_size=0.2, random_state=None):
#     '''
#     Replaced with the function balance_val_split, which returns subset of the original dataset,
#     instead of returning subsetted pandas DataFrames.
#     '''
#     targets = df[target_variable_name].tolist()
#     train_idx, val_idx = train_test_split(
#                             np.arange(len(targets)),
#                             test_size=test_size,
#                             shuffle=True,
#                             stratify=targets,
#                             random_state=random_state)
#     df_train = df.iloc[train_idx]
#     df_val = df.iloc[val_idx]

#     return df_train, df_val, train_idx, val_idx

# %%
def balance_val_split(dataset, target_variable_name, test_size=0.2, random_state=None):
    # targets = np.array(dataset.targets)
    targets = np.array(dataset.df[target_variable_name])
    train_idx, val_idx = train_test_split(
                            np.arange(len(targets)),
                            test_size=test_size,
                            shuffle=True,
                            stratify=targets,
                            random_state=random_state)
    train_dataset = Subset(dataset, indices=train_idx)
    val_dataset = Subset(dataset, indices=val_idx)
    return train_dataset, val_dataset

# %%
'''
class Subset(Dataset[T_co]):
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
'''

# %%
def get_stratified_sampler_for_subset(subset, target_variable_name):
    ''' Same as get_stratified_sampler, but for PyTorch's Subset class object. '''
    superset = subset.dataset.df
    indices  = subset.indices

    assert isinstance(superset, pd.DataFrame)   # Check this for debugging
    assert isinstance(indices, np.ndarray)      # Check this for debugging

    target = superset.iloc[indices][target_variable_name]

    class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)])

    weight = 1./class_sample_count
    assert(isinstance(weight, np.ndarray))
    print('weight: ', weight)
    samples_weight = np.array([weight[t] for t in target.tolist()]) # target needs to be int, not str
    samples_weight = torch.from_numpy(samples_weight).double()
    assert(isinstance(samples_weight, torch.Tensor))
    print('samples_weight: ', samples_weight)

    sampler = WeightedRandomSampler(weights=samples_weight, 
                                    num_samples=len(samples_weight), 
                                    replacement=False)
    return sampler, samples_weight

# %%
def get_stratified_sampler(df, target_variable_name):
    ''' Get sampler object to be used in PyTorch's DataLoader() object.
        Sampler contains the probabilities assigned to the samples.
        To be used to generate batches in DataLoader.
        sampler: torch.utils.data.sampler.WeightedRandomSampler
        samples_weight: torch.Tensor'''
    
    target = df[target_variable_name] # Should contain integers (will be used as index)

    class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)])

    weight = 1./class_sample_count
    assert(isinstance(weight, np.ndarray))
    print('weight: ', weight)
    samples_weight = np.array([weight[t] for t in target.tolist()]) # target needs to be int, not str
    samples_weight = torch.from_numpy(samples_weight).double()
    assert(isinstance(samples_weight, torch.Tensor))
    print('samples_weight: ', samples_weight)

    sampler = WeightedRandomSampler(weights=samples_weight, 
                                    num_samples=len(samples_weight), 
                                    replacement=False)
    return sampler, samples_weight

# %%
def get_dataloaders_train_val(csv_fullpath, target_variable_name, transforms, target_size, batch_size, num_workers,
                              test_size=0.2, random_state=None):
    ''' Rewritten version (define the dataset first and subset using indices, instead of creating two separate datasets) '''

    trainval_dataset = SorghumDataset(csv_fullpath=csv_fullpath,
                               transform=transforms,
                               target_size=target_size,
                               testset=False)

    dataset_train, dataset_val = balance_val_split(trainval_dataset, 'cultivar', 
                                        test_size=test_size, random_state=random_state)

    sampler_train = get_stratified_sampler_for_subset(dataset_train, target_variable_name)
    sampler_val   = get_stratified_sampler_for_subset(dataset_val, target_variable_name)

    train_loader = DataLoader(dataset       = dataset_train,
                              batch_size    = batch_size, 
                              shuffle       = False,  # sampler option is mutually exclusive with shuffle
                              num_workers   = num_workers,
                              persistent_workers = True,
                              sampler       = sampler_train)

    val_loader = DataLoader(dataset         = dataset_val, 
                            batch_size      = batch_size, 
                            shuffle         = False,  # sampler option is mutually exclusive with shuffle
                            num_workers     = num_workers,
                            persistent_workers = True,
                            sampler         = sampler_val)

    ''' Original Version '''
    # df_train, df_val, _, _ = \
    #     stratified_split_train_val(df, target_variable_name, test_size, random_state=random_state)

    # sampler_train, _ = get_stratified_sampler(df_train, target_variable_name)
    # sampler_val,   _ = get_stratified_sampler(df_val, target_variable_name)

    # dataset_train = SorghumDataset(df          = df_train, 
    #                                transform   = transforms, 
    #                                target_size = target_size, 
    #                                testset     = False)
    # dataset_val   = SorghumDataset(df          = df_val, 
    #                                transform   = transforms, 
    #                                target_size = target_size, 
    #                                testset     = False)

    # train_loader = DataLoader(dataset       = dataset_train,
    #                           batch_size    = batch_size, 
    #                           shuffle       = False,  # sampler option is mutually exclusive with shuffle
    #                           num_workers   = num_workers,
    #                           persistent_workers = True,
    #                           sampler       = sampler_train)

    # val_loader = DataLoader(dataset         = dataset_val, 
    #                         batch_size      = batch_size, 
    #                         shuffle         = False,  # sampler option is mutually exclusive with shuffle
    #                         num_workers     = num_workers,
    #                         persistent_workers = True,
    #                         sampler         = sampler_val)

    return train_loader, val_loader, dataset_train, dataset_val

# %%
if __name__=='__main__':
    # ==========================================================================================================
    # # Test stratified_split_train_val function
    # df = pd.read_csv('../../../dataset/sorghum/train_cultivar_mapping.csv')

    # df_train, df_val, train_idx, val_idx = stratified_split_train_val(
    #                                             df=df, 
    #                                             target_variable_name='cultivar', 
    #                                             test_size=0.2)
    # df_train.groupby('cultivar').count().plot.bar()
    # plt.show()
    # df_val.groupby('cultivar').count().plot.bar()
    # plt.show()

    # ==========================================================================================================
    # # Test get_stratified_sampler function
    # print(os.getcwd())
    # df = pd.read_csv('../../../dataset/sorghum/train_cultivar_mapping.csv')
    # print(df)
    # df['cultivar'] = [CULTIVAR_LABELS_ALT[cultivar] for cultivar in df['cultivar']] # Convert cultivar string to indices
    # print(df)

    # sampler, samples_weight = get_stratified_sampler(df=df, target_variable_name='cultivar')
    # print('Samples Weight: ', samples_weight)

    # ==========================================================================================================
    # # Test get_dataloaders_train_val
    # df = pd.read_csv('../../../dataset/sorghum/train_cultivar_mapping.csv')
    # df['cultivar'] = [CULTIVAR_LABELS_ALT[cultivar] for cultivar in df['cultivar']] # Convert cultivar string to indices
    # train_loader, val_loader = get_dataloaders_train_val(
    #     df = df,
    #     target_variable_name='cultivar',
    #     transforms=None,
    #     target_size=299,
    #     batch_size=32,
    #     num_workers=4)

    # print(train_loader)
    # print(val_loader)

    # ==========================================================================================================
    # dataset = SorghumDataset(csv_fullpath='/home/brian/dataset/sorghum/train_cultivar_mapping.csv',
    #                          dataset_root='/home/brian/dataset/sorghum',
    #                          transform=None,
    #                          target_size=299,
    #                          testset=False)

    # train_dataset, val_dataset = balance_val_split(dataset, 'cultivar')

    train_loader, val_loader, dataset_train, dataset_val = \
        get_dataloaders_train_val(csv_fullpath = '/home/brian/dataset/sorghum/train_cultivar_mapping.csv',
                              target_variable_name='cultivar_indx',
                              transforms = None,
                              target_size=299,
                              batch_size=32,
                              num_workers=4)


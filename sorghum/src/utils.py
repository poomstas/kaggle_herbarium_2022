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
from constants import CULTIVAR_LABELS_ALT

# %%
def stratified_split_train_val(df, target_variable_name, test_size=0.2, random_state=None):
    targets = df[target_variable_name].tolist()
    train_idx, val_idx = train_test_split(
                            np.arange(len(targets)),
                            test_size=test_size,
                            shuffle=True,
                            stratify=targets,
                            random_state=random_state)
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]

    return df_train, df_val, train_idx, val_idx

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
    samples_weight = np.array([weight[t] for t in target.tolist()])
    samples_weight = torch.from_numpy(samples_weight).double()
    assert(isinstance(samples_weight, torch.Tensor))
    print('samples_weight: ', samples_weight)

    sampler = WeightedRandomSampler(weights=samples_weight, 
                                    num_samples=len(samples_weight), 
                                    replacement=False)
    return sampler, samples_weight

# %%
def get_dataloaders_train_val(df, target_variable_name, transforms, target_size, batch_size, num_workers,
                              test_size=0.2, random_state=None):
    df_train, df_val, _, _ = \
        stratified_split_train_val(df, target_variable_name, test_size, random_state=random_state)

    sampler_train, _ = get_stratified_sampler(df_train, target_variable_name)
    sampler_val,   _ = get_stratified_sampler(df_val, target_variable_name)

    dataset_train = SorghumDataset( df          = df_train, 
                                    transform   = transforms, 
                                    target_size = target_size, 
                                    testset     = False)
    dataset_val = SorghumDataset(   df          = df_val, 
                                    transform   = transforms, 
                                    target_size = target_size, 
                                    testset     = False)

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

    return train_loader, val_loader

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
    df = pd.read_csv('../../../dataset/sorghum/train_cultivar_mapping.csv')
    df['cultivar'] = [CULTIVAR_LABELS_ALT[cultivar] for cultivar in df['cultivar']] # Convert cultivar string to indices
    train_loader, val_loader = get_dataloaders_train_val(
        df = df,
        target_variable_name='cultivar',
        transforms=None,
        target_size=299,
        batch_size=32,
        num_workers=4)

    print(train_loader)
    print(val_loader)

# %%
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from constants import CULTIVAR_LABELS, CULTIVAR_LABELS_ALT

# %%
def stratified_split_train_val(df, target_variable_name):
    return

# %%
def get_stratified_sampler(df, target_variable_name):
    ''' Get sampler object to be used in PyTorch's DataLoader() object.
        Sampler contains the probabilities assigned to the samples.
        sampler: torch.utils.data.sampler.WeightedRandomSampler
        samples_weight: torch.Tensor'''
    
    target = df[target_variable_name] # Should contain integers (will be used as index)

    class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)])

    weight = 1./class_sample_count
    samples_weight = np.array([weight[t] for t in target.tolist()])
    samples_weight = torch.from_numpy(samples_weight).double()

    sampler = WeightedRandomSampler(weights=samples_weight, 
                                    num_samples=len(samples_weight), 
                                    replacement=False)
    return sampler, samples_weight

# %%
def get_dataloaders_train_val(df, target_variable_name, train_val_split=[0.8, 0.2]):
    return

# %%
if __name__=='__main__':
    # Test get_stratified_sampler function
    print(os.getcwd())
    df = pd.read_csv('../../../dataset/sorghum/train_cultivar_mapping.csv')
    print(df)
    # Convert cultivar string to indices
    df['cultivar'] = [CULTIVAR_LABELS_ALT[cultivar] for cultivar in df['cultivar']]
    print(df)

    sampler, samples_weight = get_stratified_sampler(df=df, target_variable_name='cultivar')
    print('Samples Weight: ', samples_weight)

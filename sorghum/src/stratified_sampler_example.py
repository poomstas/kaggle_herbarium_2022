# %%
'''
Example to show how I could implement stratified sampling in PyTorch to address class imbalance problem.
Study this and implement it in the project. 
https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264
'''

# %% Imports
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

# %% Some parameters for this example
numDataPoints = 1000
data_dim = 5
bs = 100

# %% Generating an Example Training Data with Class Imbalance
# Create dummy data with class imbalance 9 to 1
data = torch.FloatTensor(numDataPoints, data_dim)
target = np.hstack((np.zeros(int(numDataPoints * 0.9), dtype=np.int32),
                    np.ones(int(numDataPoints * 0.1), dtype=np.int32)))

print("Training data's target 0-1 ratio: {}/{}".format(
    len(np.where(target == 0)[0]), len(np.where(target == 1)[0])))

# %% Calculating the Sample Probability for Each Sample and Defining the Dataset (Make this a class obj)
class_sample_count = np.array(
    [len(np.where(target == t)[0]) for t in np.unique(target)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in target]) # the prob. of each data point getting sampled by sampler

samples_weight = torch.from_numpy(samples_weight)
samples_weight = samples_weight.double() # torch.Tensor of size [1000]
sampler = WeightedRandomSampler(weights=samples_weight, 
                                num_samples=len(samples_weight), 
                                replacement=False)

target = torch.from_numpy(target).long()
train_dataset = torch.utils.data.TensorDataset(data, target)

# %% Define DataLoader
train_loader = DataLoader(
    train_dataset, batch_size=bs, num_workers=1, sampler=sampler)

# %% Using the DataLoader
for i, (data, target) in enumerate(train_loader):
    print("batch index {}, 0/1: {}/{}".format(
        i,
        len(np.where(target.numpy() == 0)[0]),
        len(np.where(target.numpy() == 1)[0])))

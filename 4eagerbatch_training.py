"""
epoch : 1 forward and backward pass of ALL training samples

batch_size : number of training samples in ONE forward and backward pass

number of iterations : number of passes per epoch , each pass using [batch_size] number of samples = number of passes needed to accomodate all training samples using batches of [batch_size]

e.g. 100 samples, batch_size=20 --> 100/20 = 5 iterations per epoch
"""

import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset,DataLoader
import math
import pandas as pd
class WineDataset(Dataset):
    def __init__(self,txtkwargs=dict(fname='./data/wine.csv',delimiter=',',dtype=np.float32,skiprows=1)):
        # data loading
        xy = np.loadtxt(**txtkwargs) # This doesnt solve the problem of the csv not fitting inside memory
        self.x = torch.from_numpy(xy[:,1:])
        self.y = torch.from_numpy(xy[:,[0]]) # shape = nsamples,1
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        # get a particular sample by index
        return self.x[index],self.y[index]
    
    def __len__(self):
        # get full length of dataset
        return self.n_samples

# Error at next(dataiter):
# Solution1 : Put everything except the WineDataset class inside and if __name__ == '__main__'
# Or Solution2 : set num_workders of DataLoader to 0
# It has something to do with Windows
"""
https://stackoverflow.com/questions/70551454/torch-dataloader-for-large-csv-file-incremental-loading
https://stackoverflow.com/questions/60101168/pytorch-runtimeerror-dataloader-worker-pids-15332-exited-unexpectedly/60101662#60101662
https://stackoverflow.com/questions/845058/how-to-get-line-count-of-a-large-file-cheaply-in-python/68385697#68385697

On windows Aneesh Cherian's solution works well for notebooks (IPython). But if you want to use num_workers>0 you should avoid interpreters like IPython and put the dataload in if __name__ == '__main__:. Also, with persistent_workers=True the dataload appears to be faster on windows if num_workers>0.

More information can be found in this thread: https://github.com/pytorch/pytorch/issues/12831
"""


dataset = WineDataset()
batch_size = 4

dataloader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0)
dataiter = iter(dataloader)
data = next(dataiter)

features,labels = data

n_epochs = 2
total_samples = len(dataset)
n_iterations =math.ceil(total_samples/batch_size)

for epoch in range(n_epochs):
    for i, (inputs,labels) in enumerate(dataloader):
        # Forward
        if i % (n_iterations//10)==0:
            print(f'epoch:{epoch+1}/{n_epochs}, step:{i+1}/{n_iterations}, inputs:{inputs.shape}')

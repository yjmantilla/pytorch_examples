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

"""
https://stackoverflow.com/questions/70551454/torch-dataloader-for-large-csv-file-incremental-loading
https://stackoverflow.com/questions/60101168/pytorch-runtimeerror-dataloader-worker-pids-15332-exited-unexpectedly/60101662#60101662
https://stackoverflow.com/questions/845058/how-to-get-line-count-of-a-large-file-cheaply-in-python/68385697#68385697

On windows Aneesh Cherian's solution works well for notebooks (IPython). But if you want to use num_workers>0 you should avoid interpreters like IPython and put the dataload in if __name__ == '__main__:. Also, with persistent_workers=True the dataload appears to be faster on windows if num_workers>0.

More information can be found in this thread: https://github.com/pytorch/pytorch/issues/12831
"""

def rawcount(filename):
    f = open(filename, 'rb')
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.raw.read

    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b'\n')
        buf = read_f(buf_size)
    f.close()
    return lines +1 # first line didnt have a preceding newline char
# Create Dataset
class CSVDataset(Dataset):
    def __init__(self, path, chunksize, nb_samples,skip_header=True):
        self.path = path
        self.chunksize = chunksize
        self.len = math.ceil(nb_samples / self.chunksize)
        self.skip_header = skip_header

    def __getitem__(self, index):
        skip = index * self.chunksize + int(self.skip_header)-1
        if skip <0:
            skip = None
        xy = next(
            pd.read_csv(
                self.path,
                skiprows=skip,  #+1, since we skip the header
                chunksize=self.chunksize))
        xy = torch.from_numpy(xy.values)
        return xy[:,1:],xy[:,[0]]

    def __len__(self):
        return self.len

filepath = './data/wine.csv'
batch_size = 4
samples = rawcount(filepath)-1 # skip header
dataset = CSVDataset(filepath,batch_size,samples,True)

dataiter = iter(dataset)
data = next(dataiter)
features,labels = data

n_epochs = 2
total_samples = samples
n_iterations =math.ceil(total_samples/batch_size)

for epoch in range(n_epochs):
    for i in range(len(dataset)): #enumerate(dataset):  not sure why enumerate iterates more than the len?
        inputs,labels=dataset[i]
        # Forward
        if i % (n_iterations//10)==0:
            print(f'epoch:{epoch+1}/{n_epochs}, step:{i+1}/{n_iterations}, inputs:{inputs.shape}')

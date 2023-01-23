import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset,DataLoader
import math
import pandas as pd
class WineDataset(Dataset):
    def __init__(self,txtkwargs=dict(fname='./data/wine.csv',delimiter=',',dtype=np.float32,skiprows=1),transform=None):
        # data loading
        xy = np.loadtxt(**txtkwargs) # This doesnt solve the problem of the csv not fitting inside memory
        self.x = xy[:,1:]
        self.y = xy[:,[0]] # shape = nsamples,1
        self.n_samples = xy.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        # get a particular sample by index
        sample = self.x[index],self.y[index]

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        # get full length of dataset
        return self.n_samples


class ToTensor:
    def __call__(self, sample):
        inputs,targets = sample
        return torch.from_numpy(inputs),torch.from_numpy(targets)

class MulTransform:
    def __init__(self,factor):
        self.factor = factor
    def __call__(self, sample):
        inputs,targets = sample
        return inputs*self.factor,targets


dataset = WineDataset(transform=ToTensor())
print('attribute x type',type(dataset.x))
print('get item return type',type(dataset[0][0]))
print('attribute x ',dataset.x)
print('get item ',dataset[0][0])

dataset = WineDataset(transform=MulTransform(10))
print('attribute x type',type(dataset.x))
print('get item return type',type(dataset[0][0]))
print('attribute x ',dataset.x)
print('get item ',dataset[0][0])

composed = torchvision.transforms.Compose([ToTensor(),MulTransform(10)])
dataset =WineDataset(transform=composed)
print('attribute x type',type(dataset.x))
print('get item return type',type(dataset[0][0]))
print('attribute x ',dataset.x)
print('get item ',dataset[0][0])

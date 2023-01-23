# A neural network without activation functions is basically a stacked linear regression model
# Non linear transformations provide more complexity

# Activations FUns

# Step       --> Not used in practice
# Sigmoid    --> Useful for last layer of binary classification
# TanH       --> Used in hidden layers
# ReLU       --> Default Value for Hidden Layers. Most popular choice. Use this if you dont know.
# Leaky ReLU --> Improved ReLU that tries to solve vanishing gradient problem
# SoftMax    --> Outputs between 0 and 1 interpreted as probabilities. Good for the last layer of a Multiclass Classification Problem

import torch
import torch.nn as nn
# Some activations are only on nn.functional
import torch.nn.functional as F 

# F.leaky_relu in example


# 2 options for ACtivations

# Option 1
class NeuralNet1(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(NeuralNet1,self).__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

# Option 2
# Use activation functions directly in forward pass
class NeuralNet2(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(NeuralNet2,self).__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,1)
    def forward(self,x):
        # Using only torch API
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out

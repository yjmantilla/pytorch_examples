import torch
import torch.nn as nn
import numpy as np

# Softmax : The idea is to get probabilities at the output

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

x = np.array([2.0,1.0,0.1])
outputs= softmax(x)
print('softmax:',outputs)

x = torch.from_numpy(x)
outputs = torch.softmax(x,dim=0)
print('softmax:',outputs)

# Cross Entropy Loss
# Measures performance when output is probabilistic, can be used in multiclass classification
# Labels must be one-hot encoded [1,0,0], notice this is equivalent to certainty of the correct class

def cross_entropy(actual,predicted):
    loss = -1*np.sum(actual*np.log(predicted))
    return loss#/float(predicted.shape[0])--> normalization if needed

# y must be one--hot

# class 0: [1,0,0]
# class 2: [0,0,1]
Y = np.array([1,0,0])
# y_pred is probabilities

Y_pred_good = np.array([0.7,0.2,0.1]) # note it sums to 1
Y_pred_bad = np.array([ 0.1,0.3,0.6])

lgood = cross_entropy(Y,Y_pred_good)
lbad  = cross_entropy(Y,Y_pred_bad)

print('good l',lgood)
print('bad l ',lbad)


## In pytorch

## nn.CrossEntropyLoss already applies nn.LogSoftmax + nn.NLLLoss (negative log likelihood loss)
# So we dont need to have Softmax in the last layer

# Y are class labels, not one-hot
# Y_pred has raw scores (logits)--> is not softmax

loss = nn.CrossEntropyLoss()
Y = torch.tensor([0])
# nsamples x nclasses = 1x3
Y_pred_good = torch.tensor([[2.0,1.0,0.1]]) # logits or raw scores
Y_pred_bad =  torch.tensor([[0.5,2.0,0.3]]) # logits or raw scores

lgood = loss(Y_pred_good,Y)
lbad  = loss(Y_pred_bad,Y)

print('good,bad')
print(lgood.item(),lbad.item())

_,pred_good=torch.max(Y_pred_good,1)
_,pred_bad =torch.max(Y_pred_bad,1)

# nsamples x nclasses = 3x3

Y = torch.tensor([2,0,1])
Y_pred_good = torch.tensor([[0.1,1.0,2.1],[2.0,1.0,0.1],[0.1,3.0,0.1]]) # logits or raw scores
Y_pred_bad =  torch.tensor([[2.1,1.0,0.1],[0.1,1.0,2.1],[0.1,3.0,0.1]]) # logits or raw scores

lgood = loss(Y_pred_good,Y)
lbad  = loss(Y_pred_bad,Y)

print('good,bad')
print(lgood.item(),lbad.item())

_,pred_good=torch.max(Y_pred_good,1)
_,pred_bad =torch.max(Y_pred_bad,1)

import numpy as np
import copy


# Data
k = 2
X = np.arange(1,5,1)
Y = copy.deepcopy(k*X)

print(X,Y)
# Parameters Init
w = 0

# Forward Model
def forward(x):
    return w*x

# Objective
def loss(y,y_ref):
    return np.mean((y-y_ref)**2)

def gradient(X,Y_pred,Y_ref):
    return np.mean(2*(Y_pred-Y_ref)*X)

# Training

n_epochs = 10
lrate = 0.01

for i in range(n_epochs):
    Y_pred = forward(X)
    L = loss(Y_pred,Y)
    grad = gradient(X,Y_pred,Y)
    w-=grad*lrate
    if i % 1==0:
        print(f'epoch:{i+1},w:{w:.3f}, l:{L:.8f}')

print(f'w_theoretical:{k},w_found:{w}')
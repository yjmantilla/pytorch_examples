# 1) Design model (input, output, forward pass with different layers)
# 2) Construct loss and optimizer
# 3) Training loop
#       - Forward = compute prediction and loss
#       - Backward = compute gradients
#       - Update weights
import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
from sklearn import datasets

CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Data
X,Y = datasets.make_regression(n_samples=100,n_features=1,noise=10,random_state=43)
X = torch.from_numpy(X.astype(np.float32))
Y = torch.from_numpy(Y.astype(np.float32))[:,None]
n_samples,n_features = X.shape
print(X,Y)

# Parameters Init
w = torch.tensor(0.0,dtype=torch.float32,requires_grad=True)

# Forward Model
input_size = X.shape[-1]
output_size = Y.shape[-1]

model = torch.nn.Linear(input_size,output_size)

# Custom model example

class CustomLinearRegression(torch.nn.Module):
    def __init__(self,input_dim,output_dim) -> None:
        super(CustomLinearRegression,self).__init__()
        #define layers
        self.lin = torch.nn.Linear(input_dim,output_dim)
    
    def forward(self,x):
        return self.lin(x)

model = CustomLinearRegression(input_size,output_size)

# GPU
if CUDA :
    X = X.to(device)
    Y = Y.to(device)
    w = w.to(device)
    model.to(device)

n_epochs = 10000
lrate = 0.001

loss = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=lrate)
# Training


for i in range(n_epochs):

    # prediction (forward)
    Y_pred = model(X)

    # loss-objective
    L = loss(Y,Y_pred)

    # backward pass to gradients
    L.backward()

    # update weights
    optimizer.step()

    optimizer.zero_grad()
    if i % (n_epochs//10)==0:
        print(f'epoch:{i+1},params:{[x.item() for x in model.parameters()]}, l:{L:.8f}')

print(f'w_found:{[x.item() for x in model.parameters()]}')

predictions = model(X).detach().cpu().numpy()

plt.plot(X.cpu().numpy(),Y.cpu().numpy(),'ro')
plt.plot(X.cpu().numpy(),predictions,'b')
plt.show()
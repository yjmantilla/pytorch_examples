# 1) Design model (input, output, forward pass with different layers)
# 2) Construct loss and optimizer
# 3) Training loop
#       - Forward = compute prediction and loss
#       - Backward = compute gradients
#       - Update weights
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# 1. Data
bc = datasets.load_breast_cancer()
X,Y = bc.data,bc.target
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=43)

# Preprocessing

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors 
X_train = torch.from_numpy(X_train.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))[:,None]
X_test =  torch.from_numpy(X_test.astype(np.float32))
Y_test =  torch.from_numpy(Y_test.astype(np.float32))[:,None]

n_samples,n_features = X.shape
print('samples',n_samples,'features',n_features)

# 2. Forward Model

# f=w*x+b with sigmoid function

input_size = X_train.shape[-1]
output_size = Y_train.shape[-1]

# Custom model example

class LogisticRegression(torch.nn.Module):
    def __init__(self,input_dim,output_dim) -> None:
        super(LogisticRegression,self).__init__()
        #define layers
        self.linear = torch.nn.Linear(input_dim,output_dim)
        self.sigmoid = torch.sigmoid
    def forward(self,x):
        return self.sigmoid(self.linear(x))

model = LogisticRegression(input_size,output_size)

# 3. GPU
if CUDA :
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)
    model.to(device)

# 4. Training Scheme

n_epochs = 1000
lrate = 0.001
objective = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=lrate)

# 5. Training Loop

for i in range(n_epochs):

    # Forward Pass (Prediction)
    Y_pred = model(X_train)

    # Objective
    J = objective(Y_pred,Y_train)

    # Backward Pass (Gradient Descend)
    J.backward()

    # Updates
    optimizer.step()
    optimizer.zero_grad() # Empty Gradient

    if i % (n_epochs//10)==0:
        print(f'epoch:{i}, J:{J:.8f}')


# 6. Validation (outside the computational graph)
with torch.no_grad():
    Y_pred = model(X_test)
    Y_pred_classes = Y_pred.round()
    acc = Y_pred_classes.eq(Y_test).float().mean()

    print(f'ACC:{acc:.4f}')

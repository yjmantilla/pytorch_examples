import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters

input_size = 28*28 # Input Image Size
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001
PLOT = False

# MNIST

root = './data/ignore_data'

train_dataset = torchvision.datasets.MNIST(root=root,train=True,
transform=transforms.ToTensor(),download=True)

test_dataset = torchvision.datasets.MNIST(root=root,train=False,
transform=transforms.ToTensor(),download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False) # Shuffle does not matter for validation

examples = iter(train_loader)
samples,labels = next(examples)
print(samples.shape,labels.shape)

if PLOT:
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.imshow(samples[i][0],cmap='gray')

    plt.show()

class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # dont apply softmax
        return out


model = NeuralNet(input_size,hidden_size,num_classes)
model.to(device)
# loss and optimizer

criterion = nn.CrossEntropyLoss() # Best for multiclass classification
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# Training Loop

n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for step,(images,labels) in enumerate(train_loader):
        # 100 samples, 1 channel, 28 height, 28 height
        # --> 100, 784

        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs,labels)

        # Backward
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        if step % (n_total_steps//10)==0:
            print(f'epoch:{epoch+1}/{num_epochs}, step:{step+1}/{n_total_steps}, loss:{loss.item():.4f}')


# test

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images,labels in test_loader:
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        #val,index
        _,preds = torch.max(outputs,1)
        n_samples += labels.shape[0]
        n_correct+= (preds==labels).sum().item()
    acc = 100.0*n_correct/n_samples
    print(f'acc={acc}')


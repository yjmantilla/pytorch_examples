import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


PLOT=False
# Device config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# HyperParams

num_epochs = 1
batch_size = 4
learning_rate = 0.001

# dataset has PILImage images of range [0,1]
# We transform them to Tensors of normalized range [-1,1]
transform = [transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
transform =  transforms.Compose(transform)

root = './data/ignore_data'
train_dataset = torchvision.datasets.CIFAR10(root=root,train=True,
transform=transform,download=True)
test_dataset = torchvision.datasets.CIFAR10(root=root,train=False,
transform=transform,download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False) # Shuffle does not matter for validation


classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')


def imshow(img,PLOT):
    if PLOT:
        img =img/2+0.5 # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg,(1,2,0)))
        plt.show()
dataiter = iter(train_loader)
images,labels = next(dataiter)

imshow(torchvision.utils.make_grid(images),PLOT)



# Conv Net
# convs--> (input_channels, output_channels, kernel/filter size)
# maxpool --> (kernel/filter size,stride)

kwargs ={
    'conv1':(3,6,5),
    'pool1':(2,2),
    'conv2':(6,16,5),
    'pool2':(2,2),
}

# Layer By Layer Dimensional Analysis
conv1 = nn.Conv2d(*kwargs['conv1'])
pool1 = nn.MaxPool2d(*kwargs['pool1'])
conv2 = nn.Conv2d(*kwargs['conv2'])
pool2 = nn.MaxPool2d(*kwargs['pool2'])

def get_conv_size(W,F,P,S):
    return int((W-F+2*P)/S +1)


print(images.shape)
size =get_conv_size(images.shape[2],kwargs['conv1'][-1],0,1)
x = conv1(images)
assert x.shape[-1]==size
print(x.shape,size)

size = get_conv_size(x.shape[2],kwargs['pool1'][0],0,kwargs['pool1'][1])
x = pool1(x)
assert x.shape[-1]==size
print(x.shape,size)

size =get_conv_size(x.shape[2],kwargs['conv2'][-1],0,1)
x = conv2(x)
assert x.shape[-1]==size
print(x.shape,size)

size = get_conv_size(x.shape[2],kwargs['pool2'][0],0,kwargs['pool2'][1])
x = pool2(x)
assert x.shape[-1]==size
print(x.shape,size)
final_flat_size = list(x.shape[1:])
print(final_flat_size,'--flatten-->',np.prod(final_flat_size))


kwargs['fc1']=(np.prod(final_flat_size),120) # Flatten Last Dimensions
kwargs['fc2']=(120,84)
kwargs['fc3']=(84,len(classes))



class ConvNet(nn.Module):
    def __init__(self,kwargs={}):
        super(ConvNet,self).__init__()
        self.kwargs = kwargs
        self.conv1 = nn.Conv2d(*kwargs['conv1'])
        self.pool1 = nn.MaxPool2d(*kwargs['pool1'])
        self.conv2 = nn.Conv2d(*kwargs['conv2'])
        self.pool2 = nn.MaxPool2d(*kwargs['pool2'])
        self.fc1   = nn.Linear(*kwargs['fc1'])
        self.fc2   = nn.Linear(*kwargs['fc2'])
        self.fc3   = nn.Linear(*kwargs['fc3'])
    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1,self.kwargs['fc1'][0])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = ConvNet(kwargs).to(device)

# loss and optimizer

criterion = nn.CrossEntropyLoss() # Best for multiclass classification
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# Training Loop

n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for step,(images,labels) in enumerate(train_loader):

        # origin shape: [4 (bach_size?),3 (channels),32,32] -> 4,3,1024
        # input_layer = 3 input channles, 6 output channels, why?? , 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs,labels)

        # Backward and Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % (n_total_steps//10)==0:
            print(f'epoch:{epoch+1}/{num_epochs}, step:{step+1}/{n_total_steps}, loss:{loss.item():.4f}')

print('Finished Training')
# test

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(len(classes))]
    n_class_samples = [0 for i in range(len(classes))]
    for images,labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        #val,index
        _,preds = torch.max(outputs,1)
        n_samples += labels.shape[0]
        n_correct+= (preds==labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = preds[i]
            if (label==pred):
                n_class_correct[label]+=1
            n_class_samples[label]+=1
    acc = 100.0*n_correct/n_samples
    print(f'acc={acc}')

    for i in range(len(classes)):
        acc = 100.0 * n_class_correct[i]/n_class_samples[i]
        print(f'Acc for {classes[i]}: {acc}%')



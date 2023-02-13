# Use Pretrain ResNET CNN
# ImageFolder 
# Scheduler--> Change the learning rate during dataset
# Transfer Learning

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets,models,transforms
import matplotlib.pyplot as plt
import time
import os
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean = np.array([0.485,0.456,0.406])
std = np.array([0.229,0.224,0.225])
data_transforms = {
    'train': transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
    ]),
    'val':transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
    ])

}
# Import data

data_dir = 'data/ignore_data/hymenoptera_data'
os.makedirs(data_dir,exist_ok=True)
sets = ['train','val']
image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train','val']}

dataloaders = {x:torch.utils.data.DataLoader(image_datasets[x],batch_size=4,shuffle=True,num_workers=0) for x in ['train','val']}

dataset_sizes = {x:len(image_datasets[x]) for x in ['train','val']}

class_names = image_datasets['train'].classes
print(class_names)

def train_model(model,criterion,optimizer,scheduler,num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-'*10)
        
        # Each Epoch has a training and validation phase
        for phase in ['train','val']:
            if phase=='train':
                model.train() # Set model to training mode
            else:
                model.eval()
            
            running_loss= 0.0
            running_corrects = 0

            # Iterate over data
            for inputs,labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward
                # track history only if in train
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _,preds = torch.max(outputs,1)
                    loss = criterion(outputs,labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                # statistics
                running_loss+= loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds==labels.data)
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_acc = running_corrects.double()/dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print('')
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


## Option 1: Fine-Tuning Method We train the whole model only a little bit
## 2nd option:  # only train last layer, other ones are freezed
only_last = True

# Import pretrained model
model = models.resnet18(pretrained=True)

if only_last:
    for param in model.parameters():
        param.requires_grad = False # If False, Freezes all layer, but we later replace the last


# Exchange the last fully connected layers

# get number of input features from the last layer (we want to imitate that for our new last layer)

num_ftrs = model.fc.in_features

# Create new layer
num_new_classes = 2
model.fc = nn.Linear(num_ftrs,num_new_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.001)

# Scheduler --> update learning rate
step_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1)
# Every step_size epochs the learning rate is multiplied by gamma
# So the learning rate reduces to 10% every 7 epochs here

model = train_model(model,criterion,optimizer,step_lr_scheduler,num_epochs=2)


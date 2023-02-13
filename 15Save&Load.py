import torch
import torch.nn as nn

"""

## Dictionaries
torch.save(arg,PATH)--> dictionaries, uses pickle

torch.load(PATH)

model.load_State_dict(arg)

## Complete Models
model class must be defined somewhere (lazy option), so it is not self-contained
torch.save(model, PATH)
model = torch.load(PATH)
model.eval()

## State Dictionary Model (Recommended)

torch.save(model.state_dict(),PATH)

model must be created again with parameters

model = Model(*args,**kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
"""

class Model(nn.Module):
    def __init__(self,n_input_features):
        super(Model,self).__init__()
        self.linear = nn.Linear(n_input_features,1)
    def forward(self,x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
model = Model(n_input_features=6)

# train your model...


## Lazy Method
fpath = "data/ignore_data/model_lazy.pth"
torch.save(model,fpath)
# load
model_loaded=torch.load(fpath)
model_loaded.eval() # set in evaluation method
for param in model_loaded.parameters():
    print(param)

# State Dict Method
fpath = "data/ignore_data/model_statedict.pth"
torch.save(model.state_dict(),fpath)
# load, we must create the model as we did originally
model_loaded=Model(n_input_features=6)
model_loaded.load_state_dict(torch.load(fpath))
model_loaded.eval() # set in evaluation method
for param in model_loaded.parameters():
    print(param)


# Saving a checkpoint of the model

# train your model...
fpath = "data/ignore_data/checkpoint.pth"

learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
print(model.state_dict(),optimizer.state_dict())

checkpoint = {
    "epoch":90,
    "model_state":model.state_dict(),
    "optim_state":optimizer.state_dict()
}

torch.save(checkpoint,fpath)


# Load
loaded_checkpoint = torch.load(fpath)
# set up the model and optimizer again

epoch = loaded_checkpoint['epoch']
model = Model(n_input_features=6)
optimizer=torch.optim.SGD(model.parameters(),lr=0)

model.load_state_dict(loaded_checkpoint['model_state'])
optimizer.load_state_dict(loaded_checkpoint['optim_state'])

print(optimizer.state_dict())

########## GPU

# Save on GPU
fpath = "data/ignore_data/modelgpu.pth"
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(),fpath)


# SAVE GPU, Load on CPU
device = torch.device('cpu')
# model = Model(*args,**kwargs)
model.load_state_dict(torch.load(fpath,map_location=device)) # give cpu as load device

# Save GPU, Load on GPU
device = torch.device("cuda")
model.load_state_dict(torch.load(fpath))
model.to(device)


# SAVE on CPU , load on GPU
device = torch.device("cuda")
model.to(device)
fpath = "data/ignore_data/modelcpu.pth"
torch.save(model.state_dict(),fpath)

device = torch.device('cuda')
model.load_state_dict(torch.load(fpath,map_location='cuda:0')) # Choose whatever gpu
model.to(device)





import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import random
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary
from Models import U_Net
from Data_Loader import Images_Dataset
from losses import calc_loss

train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if train_on_gpu else "cpu")
batch_size = 1
valid_size = 0.15
epoch = 10
random_seed = random.randint(1, 100)
num_workers = 0
pin_memory = True if train_on_gpu else False

model = U_Net().to(device)
summary(model, input_size=(4, 256, 256))

#######################################################
# Train
t_data = r'datadcm_folder\train\image'
l_data = r'datadcm_folder\train\label'
Training_Data = Images_Dataset(t_data, l_data)

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
num_train = len(Training_Data)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

if True: 
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=train_sampler,
                                           num_workers=num_workers, pin_memory=pin_memory)

valid_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=valid_sampler,
                                           num_workers=num_workers, pin_memory=pin_memory)

initial_lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=1e-5)


valid_loss_min = np.Inf

for i in range(epoch):

    train_loss = 0.0
    valid_loss = 0.0
    model.train()

    for x, y, fname in train_loader:
        print(x.shape,y.shape,fname)
        x, y, fname = x.to(device), y.to(device), fname
        optimizer.zero_grad()
        y_pred = model(x)
        loss = calc_loss(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)

    model.eval()
    torch.no_grad()

    for x, y, fname in valid_loader:
        x, y, fname = x.to(device), y.to(device), fname
        y_pred = model(x)
        loss = calc_loss(y_pred, y)
        valid_loss += loss.item() * x.size(0)

    train_loss /= len(train_loader.sampler)
    valid_loss /= len(valid_loader.sampler)

    print(f'Epoch: {i+1}/{epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}')

    if valid_loss <= valid_loss_min:
        print(f'Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), f'./model/Unet_epoch_{epoch}_batchsize_{batch_size}.pth')
        valid_loss_min = valid_loss

    scheduler.step()
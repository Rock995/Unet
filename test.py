import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import random
from torchsummary import summary
import matplotlib.pyplot as plt
from Models import U_Net
from Data_Loader import Images_Dataset

# Check if GPU is available
train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if train_on_gpu else "cpu")

# Basic parameters of the model
batch_size = 1
epoch = 10
num_workers = 0
pin_memory = True if train_on_gpu else False

# Set up the model
model = U_Net().to(device)
summary(model, input_size=(1, 512, 512))

# Paths for images and labels
test_image_dir = r'C:\Users\SYS2001\Desktop\Unet\data\membrane\test\image'
test_label_dir = r'C:\Users\SYS2001\Desktop\Unet\data\membrane\test\label'

Test_Data = Images_Dataset(test_image_dir, test_label_dir)

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])

test_loader = torch.utils.data.DataLoader(Test_Data, batch_size=batch_size, shuffle=False,
                                          num_workers=num_workers, pin_memory=pin_memory)

# Load model state
model.load_state_dict(torch.load(f'./model/Unet_epoch_{epoch}_batchsize_{batch_size}.pth'))

# Evaluation
model.eval()
dice_scores = []

for x, y, fname in test_loader:
    x, y, fname = x.to(device), y.to(device), fname
    
    with torch.no_grad():
        pred = model(x)
        pred = F.sigmoid(pred)
        pred_np = pred.cpu().numpy()
        y_np = y.cpu().numpy()
        
        # Convert to binary image
        binary_pred = (pred_np >= 0.5).astype(np.uint8)
    
        plt.imsave(f'./model/test_prediction_{fname}.png', binary_pred[0][0],cmap='gray')
       
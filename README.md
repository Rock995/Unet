# UNet: Convolutional Networks for Biomedical Image Segmentation

UNet is a convolutional neural network architecture designed for biomedical image segmentation.

## Installation
## Recommended Environment

It is recommended to use the following environment for running this code:

- Python 3.9
- PyTorch 1.10.1
- CUDA 11.3

- conda create -n unet python=3.9
- conda activate unet
- conda install pytorch==1.10.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch
- pip install -r requirements.txt
- git clone https://github.com/Rock995/Unet.git
- cd Unet
  
## Train and Test
You can place your dataset in the appropriate folders and update the data paths in the scripts. Then, run the training or testing scripts with the following commands:

- python train.py or python test.py

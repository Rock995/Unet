import os
from PIL import Image
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
import numpy as np


class Images_Dataset(Dataset):
    """Class for getting data as a Dict
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        sample : Dict of images and labels"""

    def __init__(self, images_dir, labels_dir):

       
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        self.fname_list=os.listdir(self.images_dir)

    def __len__(self):
        return len(self.fname_list)

    def __getitem__(self, idx):
        fname = self.fname_list[idx]

        img_path=os.path.join(self.images_dir,fname)
        gt_path=os.path.join(self.labels_dir,fname)
        image = self.read(img_path)
        label = self.read(gt_path)
        
        image = F.to_tensor(image)
        label = F.to_tensor(label)

        return image, label, fname

        # sample = {'images': image, 'labels': label}
        # return sample
    
    def read(self, image_path):
        assert (image_path is not None) and os.path.exists(image_path)
        
        try:
            if image_path.endswith('.png') or image_path.endswith('.jpg'):
                sample_meta = Image.open(image_path)
                sample_meta = np.array(sample_meta)
            else:
                raise ValueError(f"Unsupported file type: {image_path}")
        except Exception as e:
            print(f"Error reading image: {e}")
            return None
        
        return sample_meta



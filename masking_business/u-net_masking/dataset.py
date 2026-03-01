from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import numpy as np
import torch
import matplotlib.pyplot as plt
import albumentations as A
from PIL import Image
import os
import cv2
import pandas as pd

import sys
from pathlib import Path
from argparse import Namespace as _NS

parent_dir = Path().resolve().parent.parent
sys.path.insert(0, str(parent_dir))
#print(parent_dir)
from utils.dataset import UniversalDataset

#from utils.dataset import UniversalDataset



class OilDataset(UniversalDataset):
    """
    Wrapper for UniversalDataset that returns RGB and mask in format suitable for U-Net training.
    Returns: (image, mask) tuple where:
        - image: torch.Tensor of shape (3, H, W) - RGB image normalized to [0, 1]
        - mask: torch.Tensor of shape (1, H, W) - Binary mask as float [0.0, 1.0]
    """
    def __init__(self, ds_dir, df, mask_path=None, transform=None):
        
        self.df = df
        if mask_path is None:
            current_file = Path(__file__)  # dataset.py location
            root_dir = current_file.parent.parent.parent  # Go up to evoo/
            mask_path = root_dir / 'masking_business' / 'hand_masks'
        else:
            mask_path = Path(mask_path)

        super(OilDataset, self).__init__(root_dir = ds_dir, 
                                        meta = self.df, 
                                        load_hsi = False, 
                                        load_mask = True, 
                                        render_rgb = True, 
                                        augment = False, 
                                        crop_size = 512, 
                                        random_crop = False,
                                        mask_dir = mask_path,
                                        mask_suffix = '_mask.png'
                                        )
        self.transform = transform
    
    def __getitem__(self, idx):
        """
        Returns (image, mask) tuple for U-Net training.
        
        Returns:
            image: torch.Tensor of shape (3, H, W) - RGB image in [0, 1]
            mask: torch.Tensor of shape (1, H, W) - Binary mask in [0.0, 1.0]
        """
        # Get data from parent dataset
        sample = super(OilDataset, self).__getitem__(idx)
        
        # Extract RGB and mask
        image = sample['rgb']  # Already (3, H, W) tensor, normalized to [0, 1]
        mask = sample['mask']  # Already (1, H, W) tensor, boolean
        
        # Apply transforms if provided
        if self.transform is not None:
            if isinstance(self.transform, A.Compose):
                # Convert to numpy and scale to [0, 255] for albumentations
                image_np = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)  # (H, W, 3)
                mask_np = (mask.squeeze(0).numpy() * 255.0).astype(np.uint8)  # (H, W)
                
                transformed = self.transform(image=image_np, mask=mask_np)
                
                # Extract transformed image and mask
                # ToTensorV2 in transform converts image to tensor and normalizes to [0, 1]
                image = transformed['image']  # Already tensor (C, H, W) from ToTensorV2
                
                # Handle mask - could be numpy array or tensor depending on transforms
                mask_result = transformed['mask']
                if isinstance(mask_result, torch.Tensor):
                    # Already a tensor - ensure correct shape and range
                    mask = mask_result.float()
                    if len(mask.shape) == 2:
                        mask = mask.unsqueeze(0)  # Add channel dimension
                    # Normalize to [0, 1] if in [0, 255] range
                    if mask.max() > 1.0:
                        mask = mask / 255.0
                else:
                    # Numpy array, convert to tensor
                    mask = torch.from_numpy(mask_result).float()
                    if len(mask.shape) == 2:
                        mask = mask.unsqueeze(0)  # Add channel dimension
                    # Normalize to [0, 1] if in [0, 255] range
                    if mask.max() > 1.0:
                        mask = mask / 255.0
            else:
                # Custom transform - assume it handles torch tensors
                image, mask = self.transform(image, mask.float())
        else:
            # No transform - just convert mask to float
            mask = mask.float()
        
        return image, mask


    def __len__(self):
        # Return length of valid samples, not the full DataFrame
        # This ensures indices match between __len__ and __getitem__
        return len(self.valid_samples)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    current_file = Path(__file__)  # dataset.py location
    root_dir = current_file.parent.parent.parent  # Go up to evoo/
    pivot_path = root_dir / "pivot_table.xlsx"
    df =pd.read_excel(pivot_path)
    ds = OilDataset(ds_dir = 'D:\\oil\\', df = df, mask_path = 'C:\\Users\\4spam\\evoo\\masking_business\\hand_masks\\')
    print(len(ds))
    print(ds[0])
    print(ds[0][1].max(), ds[0][1].min())
    '''
    plt.figure()
    plt.imshow(ds[0][0].permute(1, 2, 0))
    plt.show()
    plt.figure()
    plt.imshow(ds[0][1].squeeze(0).float())
    plt.show()
    '''
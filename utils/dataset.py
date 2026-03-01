import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
import cv2
from torch.utils.data import Dataset
import torch
from spectral.io import envi
from PIL import Image
import random
from matplotlib import pyplot as plt
class UniversalDataset(Dataset):
    """
    Dataset class that intakes a stream of int ids data agnostic 
    outputs sample components according to the configuration
    """

    def __init__(self,
    root_dir: str,
    meta, # dataframe with all the stuff ['id', 'angle', 'container', 'volume', 'oil', 'date', 'distance']
    fields: List[str] = ['hsi', 'rgb', 'tare', 'evoo_conc', 'sunf_conc'],
    load_hsi: bool = True,
    load_mask: bool = True,
    render_rgb: bool = True,
    augment: bool = False,
    crop_size: Optional[int] = 140,
    random_crop: Optional[bool] = False,
    mask_dir: Optional[str] = None,
    mask_suffix: Optional[str] = '_mask.png',
    ):
        # rgb_r - rgb raw, rgb_a - rgb artificial aka rendered from hsi
        self.root_dir = Path(root_dir)
        self.df = meta
        self.ids = list(self.df['id'].values)
        self.fields = fields
        self.load_hsi = load_hsi
        self.load_mask = load_mask
        self.render_rgb = render_rgb
        self.augment = augment
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.mask_dir = Path(mask_dir)
        self.mask_suffix = mask_suffix
        self.sample_data = self._build_sample_paths()
        self.cache_s = {}
        #self.sample_paths = [s['sample_folder'] for s in self.sample_data]
        self.valid_samples = [s for s in self.sample_data if s['found']]
        
        if self.load_mask and self.mask_dir is None:
            print("Mask directory is not set, but load_mask is True")
            print("Setting load_mask to False")
            self.load_mask = False
        

        if len(self.valid_samples) < len(self.ids):
            missing_ids = [s['id'] for s in self.sample_data if not s['found']]
            print(f"Warning: \n\n Dataset initialized with error: \n\t{len(missing_ids)} IDs not found in dataset: \n{missing_ids}")

    def _build_sample_paths(self) -> List[Dict[str, Any]]:
        """
        Builds a list of sample paths for the dataset
        """
        sample_paths = []
        for id in self.ids:
            sample_info = self._find_sample_folder(id)
            sample_paths.append(sample_info)
        return sample_paths

    def _find_sample_folder(self, id: str) -> Dict[str, Any]:
        """
        Finds a sample folder for the given id
        """
        sample_info = {
            'id': id,
            'found': False,
            'sample_folder': None,
            'date_folder': None,
            'hdr_path': None,
            'raw_path': None,
            'rgb_a_path': None,
            'rgb_r_path': None,
            'mask_path': None,
        }


        folders_to_search = [d for d in self.root_dir.iterdir() if d.is_dir()]
        for folder in folders_to_search:
            sample_folder = folder / f'{id}'
            if sample_folder.exists() and sample_folder.is_dir():
                sample_info['found'] = True
                sample_info['sample_folder'] = sample_folder
                sample_info['date_folder'] = folder.name
                sample_info['hdr_path'] = sample_folder / 'results' / f'REFLECTANCE_{id}.hdr'
                sample_info['raw_path'] = sample_folder / 'results' / f'REFLECTANCE_{id}.dat'
                sample_info['rgb_a_path'] = sample_folder / 'results' / f'REFLECTANCE_{id}.png'
                sample_info['rgb_r_path'] = sample_folder / 'results' / f'RGBVIEWFINDER_{id}.png'
                break

        if self.load_mask:

            mask_path = self._find_mask(id)
            sample_info['mask_path'] = mask_path

        return sample_info

    def _find_mask(self, id: str) -> Optional[Path]:
        """
        Finds a mask for the given id
        """
        if self.mask_dir:
            mask_path = self.mask_dir / f'{id}{self.mask_suffix}'
            if mask_path.exists():
                return mask_path
        return None

    def _load_envi(self, sample_info: Dict[str, Any]) -> torch.Tensor:
        """
        Loads cube via envi file
        """
        try:
            cube_link = envi.open(sample_info['hdr_path'], sample_info['raw_path'])
        except Exception as e:
            print(f"Error loading envi file: {e}")
            print(f"Sample info: {sample_info}")
            1/0
            return None

        cube = cube_link.load()
        return torch.from_numpy(cube.astype('float32')).permute(2, 0, 1).contiguous()

    def _load_artificial_rgb(self, sample_info: Dict[str, Any]) -> torch.Tensor:
        rgb = Image.open(str(sample_info['rgb_a_path'])).convert("RGB")
        rgb = torch.from_numpy(np.array(rgb)).permute(2, 0, 1) / 255.0 # (3,H,W)
        rgb = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        rgb = torch.rot90(rgb, 1, dims=[1, 2])
        return rgb


    def _load_raw_rgb(self, sample_info: Dict[str, Any]) -> torch.Tensor:
        rgb = Image.open(str(sample_info['rgb_r_path'])).convert("RGB")
        rgb = torch.from_numpy(np.array(rgb)).permute(2, 0, 1) / 255.0 # (3,H,W)
        rgb = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        rgb = torch.rot90(rgb, 1, dims=[1, 2])
        print('In order to load real rgbs, perfrom cropping to 512/512 and alignment, so far get an error:')
        1/0
        return rgb

    def _load_mask(self, mask_path: Path) -> np.ndarray:
        """Load mask image."""
        try:
            mask = np.array(Image.open(mask_path).convert('L')) > 0
            '''
            print(mask.shape)
            print(mask.max())
            print(mask.min())
            plt.imshow(mask)
            plt.show()
            1/0
            '''
        except Exception as e:
            print(f"Error loading mask: {e}")
            print(f"info: {mask_path}")
            return None

        mask = torch.rot90(torch.from_numpy(mask), 1).unsqueeze(0)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        return mask

    def _apply_mask(self, data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return data * mask

    def _augment(self, H: torch.Tensor, rgb: torch.Tensor):
        if not self.augment:
            return H, rgb

        if random.random() < 0.5:
            H = torch.flip(H, dims=[2]);
            rgb = torch.flip(rgb, dims=[2])
        if random.random() < 0.5:
            H = torch.flip(H, dims=[1]);
            rgb = torch.flip(rgb, dims=[1])
        if random.random() < 0.5:

            k = random.randint(0, 3)
            H = torch.rot90(H, k, dims=[1, 2])
            rgb = torch.rot90(rgb, k, dims=[1, 2])
        return H, rgb

    def _crop(self, data: torch.Tensor) -> torch.Tensor:
        B, H, W = data.shape
        if H < self.crop_size or W < self.crop_size:
            return data
        if self.random_crop:
            x = random.randint(0, W - self.crop_size)
            y = random.randint(0, H - self.crop_size)
            data = data[:, y:y+self.crop_size, x:x+self.crop_size]
        else:
            x = (W - self.crop_size) // 2
            y = (H - self.crop_size) // 2
            data = data[:, y:y+self.crop_size, x:x+self.crop_size]
        return data

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_info = self.valid_samples[idx]
        sample_id = sample_info['id']
        result = {}
        result['id'] = sample_id
        if self.load_hsi:
            hsi = self._load_envi(sample_info)
        else:
            hsi = None

        # in this context render rgb means laod prerendered rgb from the dataset thumbnail, the one from 70, 53, 19
        if not self.render_rgb:
            rgb = self._load_raw_rgb(sample_info)
        else:
            rgb = self._load_artificial_rgb(sample_info)

        if hsi is not None:
            if sample_id not in self.cache_s.keys():
                thumb = hsi[:, ::8, ::8] # unifrom 64*64 pic to cut down on computation time
                mu = thumb.abs().mean(dim = 0)
                std = thumb.std(dim = 0, unbiased = False)
                cv = std / (mu + 1e-8)

                #eps = 1e-8
                #mu = cube.abs().mean(dim=0)
                #std = cube.std(dim=0, unbiased=False)
                #cv = std / (mu + eps)

                valid = mu > 1e-6  # valid mask
                if valid.any():
                    p = torch.quantile(mu[valid], 0.995)
                    bright = mu >= p
                    achrom = cv <= 0.08  # tweak 0.06–0.10 if needed
                    cand = (bright & achrom & valid)
                    if int(cand.sum()) >= 50:
                        s = torch.median(mu[cand]).item()
                    else:
                        s = p.item()
                    s = float(max(0.6, min(s, 2.0)))
                else:
                    s = 1.0

                self.cache_s[sample_id] = s

            hsi = hsi / (self.cache_s[sample_id] + 1e-8)
            hsi = hsi.clamp_(0.0, 1.0)
            # for reconstruction tier augmentation
            if self.augment:
                hsi, rgb = self._augment(hsi, rgb)
            if self.crop_size:
                hsi = self._crop(hsi)
            result['hsi'] = hsi

        if self.crop_size:
            rgb = self._crop(rgb)
           
        mask = self._load_mask(sample_info['mask_path'])

        sample_label = self.df.loc[self.df['id'] == sample_id, 'oil'].values[0]
        evoo_prop = None
        if sample_label == 'sunflower':
            evoo_prop = 0
        else:
            evoo_prop = int(sample_label.split('_')[1])

        result['tare'] = self.df.loc[self.df['id'] == sample_id, 'container'].values[0]
        result['evoo_conc'] = evoo_prop
        result['sunf_conc'] = 100 - evoo_prop
        result['rgb'] = rgb
        result['mask'] = mask
        result['id'] = sample_id
        return result

if __name__ == '__main__':
    import pandas as pd
    df = pd.read_excel('C:\\Users\\4spam\\evoo\\pivot_table.xlsx')
    ds = UniversalDataset(root_dir='D:\\oil\\', meta=df, mask_dir='C:\\Users\\4spam\\evoo\\masking_business\\hand_masks\\')
    print(ds[0])









































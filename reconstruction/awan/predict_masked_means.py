import os
import csv
from pathlib import Path
from typing import List, Tuple
from argparse import Namespace as _NS

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Reuse components from the training script
from normalized_dirty_awan import UniversalReconstructionDataset, AWANCleanNet

# Configuration
checkpoint_path = 'checkpoints/awan_dirty_random_crop_192p_200ch_8b_psnl_20250119_143022.pth'  # Update this path
catalog_path = ''  # Set on target machine
train_path = 'train_list.txt'
val_path = 'val_list.txt'
test_path = 'test_list.txt'
waves_file = 'fx10_wavelengths_224.npy'
css_file = 'css_fx10_204.npy'
output_csv = 'masked_predictions.csv'

def extract_paths(filename: str, catalog_path: str) -> List[Tuple[str, str, str]]:
    """Extract (hdr, dat, png) paths from list file."""
    paths: List[Tuple[str, str, str]] = []
    with open(filename, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            dat = catalog_path + line.strip()
            hdr = catalog_path + line.split('.dat')[0] + '.hdr'
            ident = line.split('/results/')[0].split('/')[-1]
            png = catalog_path + line.split('/results/')[0] + '/' + ident + '.png'
            paths.append((hdr, dat, png))
    return paths

def load_mask(mask_path: str) -> torch.Tensor:
    """Load binary mask from PNG file. Returns (H, W) tensor with 1.0 for white pixels, 0.0 for black."""
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    
    mask_img = Image.open(mask_path).convert('L')  # Convert to grayscale
    mask_array = np.array(mask_img)
    # Normalize: white (255) -> 1.0, black (0) -> 0.0
    mask = torch.from_numpy(mask_array).float() / 255.0
    # Threshold to ensure binary: > 0.5 -> 1.0, <= 0.5 -> 0.0
    mask = (mask > 0.5).float()
    return mask

def get_mask_path(png_path: str) -> str:
    """Get mask path from PNG path. Mask is named id_mask.png in same directory."""
    png_file = Path(png_path)
    id_name = png_file.stem  # Get filename without extension
    mask_path = png_file.parent / f"{id_name}_mask.png"
    return str(mask_path)

def compute_masked_mean(pred_cube: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """
    Apply mask to predicted cube and compute mean per wavelength.
    
    Args:
        pred_cube: (Bands, H, W) predicted HSI cube
        mask: (H, W) binary mask (1.0 for white, 0.0 for black)
    
    Returns:
        (Bands,) array of mean values per wavelength (only counting white pixels)
    """
    bands, h, w = pred_cube.shape
    mask_3d = mask.unsqueeze(0).expand(bands, -1, -1)  # (Bands, H, W)
    
    # Apply mask pixelwise
    masked_cube = pred_cube * mask_3d  # (Bands, H, W)
    
    # Compute mean per band (only counting white pixels)
    mask_sum = mask.sum()  # Total number of white pixels
    if mask_sum > 0:
        means = masked_cube.sum(dim=(1, 2)) / mask_sum  # (Bands,)
    else:
        # If mask is all black, return zeros
        means = torch.zeros(bands)
    
    return means.cpu().numpy()

def main():
    # Load checkpoint and extract args
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Try to read args from checkpoint (new format)
    if 'args' in ckpt:
        print("Loading args from checkpoint metadata...")
        args_dict = ckpt['args']
        args = _NS(**args_dict)
        print(f"Loaded args: render={args.render}, channels={args.channels}, blocks={args.blocks}, use_psnl={not args.no_psnl}")
    else:
        # Fallback to filename parsing (backward compatibility)
        print("Args not found in checkpoint, parsing filename...")
        filename = Path(checkpoint_path).stem
        parts = filename.split('_')
        
        args = _NS(
            waves=waves_file,
            css=css_file,
            catalog_path=catalog_path,
            channels=200,
            blocks=8,
            no_psnl=False,
            render=False,
        )
        
        # Parse from filename
        if len(parts) > 0:
            args.render = parts[0].split('awan_')[-1] == 'clean' if 'clean' in parts[0] else False
        for p in parts:
            if 'ch' in p:
                args.channels = int(p.split('ch')[0].split('_')[-1])
            if 'b' in p and 'ch' not in p and 'nopsnl' not in p:
                args.blocks = int(p.split('b')[0].split('_')[-1])
        args.no_psnl = 'nopsnl' in filename
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load wavelengths and CSS
    waves = np.load(args.waves).astype('float32')
    css = np.load(args.css).astype('float32')
    
    # Build model
    model = AWANCleanNet(bands_out=css.shape[0], channels=args.channels, n_blocks=args.blocks, use_psnl=not args.no_psnl)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval().to(device)
    print("Model loaded successfully.")
    
    # Load all datasets
    print("\nLoading datasets...")
    train_pairs = extract_paths(train_path, catalog_path)
    val_pairs = extract_paths(val_path, catalog_path)
    test_pairs = extract_paths(test_path, catalog_path)
    
    print(f"Train: {len(train_pairs)} samples")
    print(f"Val: {len(val_pairs)} samples")
    print(f"Test: {len(test_pairs)} samples")
    
    # Create datasets (no augmentation, full size)
    train_ds = UniversalReconstructionDataset(train_pairs, waves, css, patch_size=None, augment=False, render=args.render)
    val_ds = UniversalReconstructionDataset(val_pairs, waves, css, patch_size=None, augment=False, render=args.render)
    test_ds = UniversalReconstructionDataset(test_pairs, waves, css, patch_size=None, augment=False, render=args.render)
    
    # Prepare CSV output
    num_bands = css.shape[0]
    csv_columns = ['rgb_path', 'id'] + [f'band_{i}' for i in range(num_bands)]
    
    print(f"\nStarting prediction and mask processing...")
    print(f"Output CSV: {output_csv}")
    
    all_results = []
    total_processed = 0
    
    # Process train set
    print("\nProcessing TRAIN set...")
    for idx, sample in enumerate(train_ds):
        rgb = sample['rgb'].unsqueeze(0).to(device)  # (1, 3, H, W)
        
        # Get paths for this sample
        hdr_path, dat_path, png_path = train_pairs[idx]
        mask_path = get_mask_path(png_path)
        
        # Predict
        with torch.no_grad():
            pred_cube = model(rgb).squeeze(0)  # (Bands, H, W)
        
        # Load and apply mask
        try:
            mask = load_mask(mask_path)  # (H, W)
            # Ensure mask matches prediction size
            if mask.shape != pred_cube.shape[1:]:
                # Resize mask to match prediction
                mask = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(pred_cube.shape[1], pred_cube.shape[2]),
                    mode='nearest'
                ).squeeze()
            
            # Compute masked means
            masked_means = compute_masked_mean(pred_cube, mask)
            
            # Extract ID from path
            id_name = Path(png_path).stem
            
            # Store results
            row = [png_path, id_name] + masked_means.tolist()
            all_results.append(row)
            
            total_processed += 1
            if total_processed % 10 == 0:
                print(f"  Processed {total_processed} entries...")
        
        except FileNotFoundError as e:
            print(f"  Warning: {e}, skipping sample {idx}")
            continue
    
    # Process val set
    print("\nProcessing VAL set...")
    for idx, sample in enumerate(val_ds):
        rgb = sample['rgb'].unsqueeze(0).to(device)
        hdr_path, dat_path, png_path = val_pairs[idx]
        mask_path = get_mask_path(png_path)
        
        with torch.no_grad():
            pred_cube = model(rgb).squeeze(0)
        
        try:
            mask = load_mask(mask_path)
            if mask.shape != pred_cube.shape[1:]:
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(pred_cube.shape[1], pred_cube.shape[2]),
                    mode='nearest'
                ).squeeze()
            
            masked_means = compute_masked_mean(pred_cube, mask)
            id_name = Path(png_path).stem
            row = [png_path, id_name] + masked_means.tolist()
            all_results.append(row)
            
            total_processed += 1
            if total_processed % 10 == 0:
                print(f"  Processed {total_processed} entries...")
        
        except FileNotFoundError as e:
            print(f"  Warning: {e}, skipping sample {idx}")
            continue
    
    # Process test set
    print("\nProcessing TEST set...")
    for idx, sample in enumerate(test_ds):
        rgb = sample['rgb'].unsqueeze(0).to(device)
        hdr_path, dat_path, png_path = test_pairs[idx]
        mask_path = get_mask_path(png_path)
        
        with torch.no_grad():
            pred_cube = model(rgb).squeeze(0)
        
        try:
            mask = load_mask(mask_path)
            if mask.shape != pred_cube.shape[1:]:
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(pred_cube.shape[1], pred_cube.shape[2]),
                    mode='nearest'
                ).squeeze()
            
            masked_means = compute_masked_mean(pred_cube, mask)
            id_name = Path(png_path).stem
            row = [png_path, id_name] + masked_means.tolist()
            all_results.append(row)
            
            total_processed += 1
            if total_processed % 10 == 0:
                print(f"  Processed {total_processed} entries...")
        
        except FileNotFoundError as e:
            print(f"  Warning: {e}, skipping sample {idx}")
            continue
    
    # Write CSV
    print(f"\nWriting results to {output_csv}...")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_columns)
        writer.writerows(all_results)
    
    print(f"\nDone! Processed {total_processed} entries total.")
    print(f"Results saved to {output_csv}")
    print(f"CSV contains {len(csv_columns)} columns: rgb_path, id, and {num_bands} band means")

if __name__ == "__main__":
    main()


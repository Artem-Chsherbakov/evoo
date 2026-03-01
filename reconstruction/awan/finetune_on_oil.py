from utils.dataset import UniversalDataset
from normalized_dirty_awan import *

from normalized_dirty_awan import *

import os, re, math, json, random, time
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from spectral.io import envi
from huggingface_hub import HfApi, hf_hub_download
from spectral.io import envi
from pathlib import Path
import re
from PIL import Image
from argparse import Namespace as _NS
import math, random
import numpy as np
import torch
from torch.utils.data import DataLoader
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
import pandas as pd

"""
"C:/Users/4spam/DL/proj/new_awan/awan_clean_random_crop_240p_600ch_6b_nopsnl_10.0t_20251211_001846.pth"
"""

def main():
    args = _NS(
        catalog_path='D:/oil/',  # Will be set on target machine
        cut_out = False,
        waves='fx10_wavelengths_204.npy',
        device='cuda:0',
        epochs=25,
        bs=2,
        lr=1e-5,
        patch=240,
        render=False,  # Set to False for dirty training
        center_crop=False,
        channels=600,
        blocks=6,
        no_psnl=True, # need a beefy ass machine for that
        tau=10.0,  # Only used when render=True
        workers=16,
        val_frac=0.05,
        pivot_table_path='C:/Users/4spam/evoo/pivot_table.xlsx',
        out='',  # Will be auto-generated
        pretrained_checkpoint="C:/Users/4spam/DL/proj/new_awan/awan_clean_random_crop_240p_600ch_6b_nopsnl_10.0t_20251211_001846.pth",
    )
    
    # Generate output filename
    out_string = 'awan_' + ('clean' if args.render else 'dirty')
    out_string += f'_center_crop' if args.center_crop else '_random_crop'
    out_string += f'_{args.patch}p' if args.patch else '_fp'
    out_string += f'_{args.channels}ch' + f'_{args.blocks}b'
    out_string += f'_nopsnl' if args.no_psnl else '_psnl'
    if args.render:
        out_string += f'_{args.tau}t'
    out_string += '_' + args.pretrained_checkpoint.split('_')[-1].split('.pth')[0]
    out_string += f'_oil_tuned'
    args.out = out_string


    device = torch.device(args.device)
    

    

    if os.path.exists('train_list.csv'):
        train_df = pd.read_csv('train_list.csv')
        val_df = pd.read_csv('val_list.csv')
        test_df = pd.read_csv('test_list.csv')
    else:
        df = pd.read_excel(args.pivot_table_path)
        df = df.sample(frac=1).reset_index(drop=True)
        train_df = df.head(int(len(df) * 0.8))
        val_df = df.tail(int(len(df) * 0.1))
        test_df = df.tail(int(len(df) * 0.1))
        test_df.to_csv('test_list.csv', index=False)
        val_df.to_csv('val_list.csv', index=False)
        train_df.to_csv('train_list.csv', index=False)
    
    if args.cut_out:
        train_df = train_df.head(5)
        val_df = val_df.head(5)
        test_df = test_df.head(5)

    print(len(train_df))
    print(len(val_df))
    waves = np.load(args.waves).astype('float32')   # (B,)
    print('done load')

    # Create directories for logs and checkpoints
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    train_ds = UniversalDataset(root_dir = args.catalog_path,  meta = train_df,
                                load_hsi = True,
                                load_mask = False,
                                render_rgb = True,
                                augment = True,
                                crop_size = args.patch,
                                random_crop = True,
                                )

    val_ds   = UniversalDataset(root_dir = args.catalog_path,  meta = val_df,
                                load_hsi = True,
                                load_mask = False,
                                render_rgb = True,
                                augment = True,
                                crop_size = args.patch,
                                random_crop = True,
                                )
    workers = args.workers

    if workers > 0:
        train_dl = DataLoader(
            train_ds, batch_size=args.bs, shuffle=True,
            num_workers=workers, persistent_workers=True,
            pin_memory=True, prefetch_factor=4, pin_memory_device="cuda"
        )
    else:
        train_dl = DataLoader(
            train_ds, batch_size=args.bs, shuffle=True,
            num_workers=0, pin_memory=True
        )

    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    bands_out = css.shape[0]  # or use len(waves); they should match
    # Build model with 4 input channels
    model = AWANCleanNet(
        bands_out=css.shape[0],
        channels=args.channels,
        n_blocks=args.blocks,
        use_psnl=not args.no_psnl,
        in_channels=4
    )

    if args.pretrained_checkpoint:
        model = load_weights_3_to_4(model, args.pretrained_checkpoint, device, init_strategy='random')
    model.to(device)
    model = model.float()
    # Initialize loss function based on render flag
    if args.render:
        loss_fn = CSSReprojectionLoss(waves, css, tau=args.tau).to(device)
        is_dirty = False
    else:
        loss_fn = DirtyTrainingLoss().to(device)
        is_dirty = True
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    # Initialize logging
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_history_{log_timestamp}.json"
    training_history = []

    best = math.inf
    print('done dl setup')
    total_time = time.time()
    training_start_time = datetime.now().isoformat()
    
    for ep in range(1, args.epochs+1):
        print(f"\nEpoch {ep}/{args.epochs}")
        ep_start = time.time()
        tr = train_one_epoch(model, loss_fn, opt, train_dl, device, scaler, is_dirty=is_dirty)
        ep_end = time.time()
        inf_start = time.time()
        va = evaluate(model, val_dl, device)
        inf_end = time.time()
        
        print(f"Val: RMSE={va['rmse']:.4f}  MRAE={va['mrae']:.4f}  TrainLoss={tr['loss']:.4f}")
        print(f"Time: TT = {ep_end - ep_start:.2f}s   INF = {inf_end - inf_start:.2f}s")
        
        # Log metrics to history
        log_entry = {
            'epoch': ep,
            'train': tr,
            'val': va,
            'timestamp': datetime.now().isoformat(),
            'train_time': ep_end - ep_start,
            'eval_time': inf_end - inf_start
        }
        training_history.append(log_entry)
        
        # Save logs periodically (every epoch)
        with open(log_file, 'w') as f:
            json.dump({
                'training_start': training_start_time,
                'args': vars(args),
                'history': training_history
            }, f, indent=2)
        
        # Save best checkpoint by RMSE
        if va['rmse'] < best:
            best = va['rmse']
            checkpoint_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = f"checkpoints/{args.out}_{checkpoint_timestamp}.pth"
            
            checkpoint = {
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'ep': ep,
                'best_rmse': best,
                'timestamp': datetime.now().isoformat(),
                'args': vars(args),  # Store all training arguments for easy parsing
                'val_metrics': va,
                'train_metrics': tr
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved best → {checkpoint_path}")
    
    tt = time.time() - total_time
    h = int(tt // 3600)
    m = int((tt - h*3600) // 60)
    s = int(tt - h * 3600 - m * 60)
    print("Done.")
    print(f"Total training time: {h}h {m}m {s}s")
    
    # Save final log entry
    with open(log_file, 'w') as f:
        json.dump({
            'training_start': training_start_time,
            'training_end': datetime.now().isoformat(),
            'total_time_seconds': tt,
            'args': vars(args),
            'history': training_history
        }, f, indent=2)
    
    torch.cuda.empty_cache()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()           # good practice on Windows
    # (optional) torch.multiprocessing.set_start_method("spawn", force=True)
    main()
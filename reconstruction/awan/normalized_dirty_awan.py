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
# -----------------------------
# Utility: CSS renderer (linear)
# -----------------------------
class CSSRenderer(nn.Module):
    """HSI (B,H,W) → linear RGB (3,H,W) via per-wavelength CSS and Δλ.
    Precompute weight W = Δλ[:,None] * CSS (B,3). Use same instance in Dataset and Loss.
    """

    def __init__(self, wavelengths: np.ndarray, css: np.ndarray):
        super().__init__()
        waves = torch.as_tensor(wavelengths, dtype=torch.float32)
        dl = torch.gradient(waves)[0]  # (B,)
        css_t = torch.as_tensor(css, dtype=torch.float32)  # (B,3)
        weight = (dl[:, None] * css_t)  # (B,3)
        self.register_buffer('weight', weight)  # (B,3)

    @torch.no_grad()
    def render_np(self, H_hw_b: np.ndarray) -> np.ndarray:
        """For numpy arrays on CPU. H: (H,W,B) → RGB_lin: (H,W,3)."""
        H = H_hw_b.reshape(-1, H_hw_b.shape[2])  # (HW,B)
        RGB = H @ self.weight.cpu().numpy()  # (HW,3)
        return np.clip(RGB.reshape(H_hw_b.shape[0], H_hw_b.shape[1], 3), 0.0, None)

    def forward(self, H_b_hw: torch.Tensor) -> torch.Tensor:
        """H: (B,H,W) tensor → RGB: (3,H,W)."""
        Bc, Hh, Ww = H_b_hw.shape
        HW = Hh * Ww
        H_flat = H_b_hw.reshape(Bc, -1).t()  # (HW,B)
        RGB = H_flat @ self.weight  # (HW,3)
        RGB = RGB.reshape(Hh, Ww, 3).permute(2, 0, 1)  # (3,H,W)
        return torch.clamp(RGB, min=0.0)


# -----------------------------
# Dataset: ENVI reflectance → (RGB_lin, HSI)
# -----------------------------
class UniversalReconstructionDataset(Dataset):
    def __init__(self,
                 items: List[Tuple[str, str]],  # list of (hdr_path, dat_path, png_path)
                 wavelengths: np.ndarray,
                 css: np.ndarray,
                 patch_size: Optional[int] = None, center_crop: bool = False,
                 augment: bool = True, use_remote = False, render = False,
                 ensemble_flip: bool = False, mask: bool = False, 
                 mask_path: Optional[str] = 'mask_and_rgb_apple_cultivars', # path to either folder where we keep png mask, or to the df that contains coord masks
                 mask_format: str = 'png',
                 shoot_labels: bool = False,
                 augment_rgb: bool = False,
                 augment_nm_idx: int = 153,
                 ):

        self.items = [(str(Path(h)), str(Path(d)), str(Path(p))) for (h, d, p) in items]
        self.renderer = CSSRenderer(wavelengths, css)
        self.patch = patch_size
        self.center_crop = center_crop
        self.augment = augment
        self.use_remote = use_remote
        self._render = render
        self._ensemble_flip = ensemble_flip
        self._cache = {}
        self.cache_s = {}
        self.mask = mask
        self.mask_path = mask_path
        self.shoot_labels = shoot_labels
        self.augment_rgb = augment_rgb
        self.augment_nm_idx = augment_nm_idx
        if self.mask and self.mask_format == 'cord':
            self.mask_df = pd.read_csv(mask_path)
        else:
            self.mask_df = None
        if self.use_remote:
            self.repo_id = "issai/Apples_HSI"
            self.api = HfApi()
            self.files = set(self.api.list_repo_files(repo_id=self.repo_id, repo_type="dataset"))

    def __len__(self):
        return len(self.items)

    def _resolve(self, p: str) -> str:
        """Return a local path for p. If p is remote (repo-relative), download once."""
        # If already local file, just return
        if Path(p).exists():
            return p
        if not self.use_remote:
            raise FileNotFoundError(f"Local path not found (use_remote=False): {p}")

        # Normalize to repo-relative (no leading ./)
        key = p.lstrip("./")
        if key not in self.files:
            # Common pitfall: wrong folder or Windows backslashes
            alt = key.replace("\\", "/")
            if alt in self.files:
                key = alt
            else:
                raise FileNotFoundError(f"'{key}' is not in HF repo file list. "
                                        f"Make sure you pass repo-relative paths exactly as listed by HfApi.")
        if key not in self._cache:
            self._cache[key] = hf_hub_download(repo_id=self.repo_id,
                                               repo_type="dataset",
                                               filename=key)
        return self._cache[key]

    def _random_crop(self, H: torch.Tensor, rgb: torch.Tensor, size: int):
        # H: (B,H,W), rgb: (3,H,W)
        _, Hh, Ww = H.shape
        if Hh < size or Ww < size:
            return H, rgb  # skip if small
        y = random.randint(0, Hh - size)
        x = random.randint(0, Ww - size)
        return H[:, y:y + size, x:x + size], rgb[:, y:y + size, x:x + size]

    def _center_crop(self, H: torch.Tensor, rgb: torch.Tensor, size: int):
        # H: (B,H,W), rgb: (3,H,W)
        _, Hh, Ww = H.shape
        if Hh < size or Ww < size:
            return H, rgb  # skip if small
        y = Hh // 2 - size // 2
        x = Ww // 2 - size // 2
        return H[:, y:y + size, x:x + size], rgb[:, y:y + size, x:x + size]

    def _augment(self, H: torch.Tensor, rgb: torch.Tensor):
        if not self.augment:
            return H, rgb
        # flips / 90° rotations
        if random.random() < 0.5:
            H = torch.flip(H, dims=[2]);
            rgb = torch.flip(rgb, dims=[2])
        if random.random() < 0.5:
            H = torch.flip(H, dims=[1]);
            rgb = torch.flip(rgb, dims=[1])
        if random.random() < 0.5:
            # rot 90 k times
            k = random.randint(0, 3)
            H = torch.rot90(H, k, dims=[1, 2])
            rgb = torch.rot90(rgb, k, dims=[1, 2])
        return H, rgb

    def _get_mask_path(self, png_path: str) -> str:
        """Get mask path from PNG path. Mask is in mask_and_rgb_apple_cultivars directory with <png_id>_mask.png naming."""
        png_file = Path(png_path)
        id_name = png_file.stem  # Get filename without extension (e.g., "1000" from "1000.png")
        # Mask is in mask_and_rgb_apple_cultivars directory
        mask_path = Path(self.mask_path) / f"{id_name}_mask.png"
        return str(mask_path)

    def __getitem__(self, i):
        hdr_path, dat_path, png_path = self.items[i]
        hdr_local = self._resolve(hdr_path)
        dat_local = self._resolve(dat_path)
        png_local = self._resolve(png_path)
        id = Path(png_path).stem
        #print(png_local) # C:\Users\artem.chsherbakov\Documents\applejam\Catalogs\apple_jam_granny_25_75_19_Jan\1336\1336.png
        img = envi.open(hdr_local, image=dat_local)
        cube = torch.from_numpy(img.load().astype('float32'))  # (H,W,B)
        cube = cube.permute(2, 0, 1).contiguous()  # (B,H,W)
        cultivar = png_local.split('apple_jam_')[-1].split('_')[0]
        sugar = png_local.split(cultivar + '_')[-1].split('_')[0]
        apple = png_local.split(cultivar + '_')[-1].split('_')[1]
        # hsi in original dataset has excessive reflectance for flares, so here is apparently costly per-cube mu/stdev normalization
        if hdr_path not in self.cache_s.keys():
            thumb = cube[:, ::8, ::8] # unifrom 64*64 pic
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

            self.cache_s[hdr_path] = s

        cube = cube / (self.cache_s[hdr_path] + 1e-8)
        cube = cube.clamp_(0.0, 1.0)

        if not self._render:
            rgb = Image.open(png_local).convert("RGB")
            rgb = torch.from_numpy(np.array(rgb)).permute(2, 0, 1) / 255.0 # (3,H,W)
            rgb = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
            rgb = torch.rot90(rgb, 1, dims=[1, 2])
        else:
            rgb = self.renderer(cube)  # (3,H,W) linear sRGB
        #print(f'RGB shape: {rgb.shape}')
        #print(f'HSI shape: {cube.shape}')

        if self.augment_rgb:
            nir_slice = cube[self.augment_nm_idx]
            rgb = torch.cat([rgb, nir_slice.unsqueeze(0)], dim=0)

        if self.mask:
            if self.mask_format == 'png':
                mask_path = self._resolve(self._get_mask_path(png_path))
                #print(mask_path)
                msk = np.array(Image.open(mask_path).convert('L')) > 0
                #print(msk.shape)
                #print(msk.astype(np.uint8).sum())
                msk = torch.rot90(torch.from_numpy(msk), 1).unsqueeze(0)

                '''
                plt.imshow(msk.squeeze().numpy(), cmap='gray', vmin=0, vmax=1 if msk.dtype == bool or msk.max() <= 1 else 255)
                plt.axis('off')
                plt.show()
                plt.imshow(cube[0])
                plt.axis('off')
                plt.show()
                plt.imshow(rgb.permute(1, 2, 0).numpy())
                plt.axis('off')
                plt.show()
                '''
                rgb = rgb * msk
                cube = cube * msk
                '''
                plt.imshow(cube[0])
                plt.axis('off')
                plt.show()
                plt.imshow(rgb.permute(1, 2, 0).numpy())
                plt.axis('off')
                plt.show()
                '''
            elif self.mask_format == 'cord':
                raise ValueError(f"No cord masking implemented, use png")
            else:
                raise ValueError(f"Invalid mask format: {self.mask_format}, use png or cord")

        if self.patch:
            if self.center_crop:
                cube, rgb = self._center_crop(cube, rgb, self.patch)
            else:
                cube, rgb = self._random_crop(cube, rgb, self.patch)
        # next is regular stuff
        cube, rgb = self._augment(cube, rgb)

        # Apply ensemble flip if requested (after all manipulations)
        if self._ensemble_flip:
            # Rotate 180 degrees (2 rotations of 90 degrees)
            rgb = torch.rot90(rgb, 2, dims=[1, 2])
            cube = torch.rot90(cube, 2, dims=[1, 2])
        out = {'rgb': rgb, 'hsi': cube, 'id': id}
        if self.shoot_labels:
            out['cultivar'] = cultivar
            out['sugar'] = int(sugar)
            out['apple'] = int(apple)
        return out



# -----------------------------
# Model: AWAN-like (DRAB + AWCA + optional PSNL)
# -----------------------------
class AWCA(nn.Module):
    def __init__(self, c: int, reduction: int = 16):
        super().__init__()
        m = max(1, c // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(c, m, 1), nn.ReLU(inplace=True), nn.Conv2d(m, c, 1), nn.Sigmoid()
        )

    def forward(self, x):
        w = self.mlp(self.avg(x))
        return x * w


class DRAB(nn.Module):
    """Dual Residual Attention Block (simplified):
    two conv paths (3x3 & 5x5), sum, AWCA, residual.
    """
    def __init__(self, c: int):
        super().__init__()
        self.p3 = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.PReLU(), nn.Conv2d(c, c, 3, padding=1))
        self.p5 = nn.Sequential(nn.Conv2d(c, c, 5, padding=2), nn.PReLU(), nn.Conv2d(c, c, 5, padding=2))
        self.ca = AWCA(c)

    def forward(self, x):
        y = self.p3(x) + self.p5(x)
        y = self.ca(y)
        return x + y


class SimpleNonLocal(nn.Module):
    def __init__(self, c: int, reduction: int = 8, grid: int = 3,
                 center_hw: int = 56, use_softmax: bool = False,
                 max_softmax_positions: int = 4096):
        super().__init__()
        assert grid == 3, "This implementation is set up for 3×3."
        self.grid = grid
        self.center_hw = center_hw
        self.use_softmax = use_softmax
        self.max_softmax_positions = int(max_softmax_positions)

        d = max(1, c // reduction)
        self.to_B = nn.Conv2d(c, d, 1, bias=False)
        self.to_D = nn.Conv2d(c, d, 1, bias=False)
        self.phi = nn.Conv2d(d, c, 1, bias=False)

    def _grid_slices(self, H: int, W: int):
        # Make the center cell ~center_hw × center_hw, side cells share the remainder
        def axis_slices(L: int):
            side = max(1, (L - self.center_hw) // 2)
            mid = max(1, L - 2 * side)
            starts = [0, side, side + mid]
            stops = [side, side + mid, L]
            return list(zip(starts, stops))

        ys = axis_slices(H)
        xs = axis_slices(W)
        # print(f"H {H}, W {W}, on y {ys}, on x {xs}")
        # 1/0
        return ys, xs  # lists of 3 (start, stop) pairs for y and x

    def _psnl_on_patch(self, f: torch.Tensor) -> torch.Tensor:
        # f: (B,C,h,w)
        Bsz, C, h, w = f.shape
        n = h * w
        Bf = self.to_B(f)  # (Bsz, d, h, w)
        Df = self.to_D(f)  # (Bsz, d, h, w)
        d = Bf.size(1)

        # (Bsz, n, d) with spatial flattened
        Bn = Bf.view(Bsz, d, n).permute(0, 2, 1).contiguous()
        Dn = Df.view(Bsz, d, n).permute(0, 2, 1).contiguous()

        # Centering => second-order stats (X = B I B^T)
        mu = Bn.mean(dim=1, keepdim=True)  # (Bsz, 1, d)
        Bc = Bn - mu  # (Bsz, n, d)

        if self.use_softmax and n <= self.max_softmax_positions:
            # Paper-style: X = Bc @ Bc^T, A = softmax(X), U = A @ D  (Eq. 4–5)
            X = torch.bmm(Bc, Bc.transpose(1, 2)) / (d ** 0.5)  # (Bsz, n, n)
            A = torch.softmax(X, dim=-1)
            U = torch.bmm(A, Dn)  # (Bsz, n, d)
        else:
            # Linearized 2nd-order (memory-safe): U = Bc @ (Bc^T @ Dn) / n
            # avoids building (n×n).
            T = torch.bmm(Bc.transpose(1, 2), Dn) / float(n)  # (Bsz, d, d)
            U = torch.bmm(Bc, T)  # (Bsz, n, d)

        U = U.permute(0, 2, 1).contiguous().view(Bsz, d, h, w)  # (Bsz, d, h, w)
        return self.phi(U) + f  # residual (Eq. 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Bsz, C, H, W = x.shape
        # print(x.shape)
        y = torch.empty_like(x)
        ys, xs = self._grid_slices(H, W)

        for iy, (y0, y1) in enumerate(ys):
            for ix, (x0, x1) in enumerate(xs):
                patch = x[:, :, y0:y1, x0:x1]
                y_patch = self._psnl_on_patch(patch)
                y[:, :, y0:y1, x0:x1] = y_patch

        return y


class AWANCleanNet(nn.Module):
    def __init__(self, in_channels: int,bands_out: int = 3, channels: int = 200, n_blocks: int = 8, use_psnl: bool = True):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, channels, 3, padding=1), nn.PReLU())
        self.body = nn.Sequential(*[DRAB(channels) for _ in range(n_blocks)])
        self.psnl = SimpleNonLocal(channels) if use_psnl else nn.Identity()
        self.head = nn.Conv2d(channels, bands_out, 3, padding=1)

    def forward(self, x_rgb):
        f0 = self.stem(x_rgb)
        f = self.body(f0)
        f = self.psnl(f)
        f = f + f0  # global residual
        out = self.head(f)
        return out  # (B_out,H,W)


def load_weights_3_to_4(model, checkpoint_path, device, init_strategy: str = 'random'):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    pretrained = ckpt['model']
    model_state = model.state_dict()

    stem_key = 'stem.0.weight'
    stem_bias_key = 'stem.0.bias'

    if stem_key in pretrained:
        pretrained_stem = pretrained[stem_key]
        current_stem = model_state[stem_key]
        
        # ✅ Ensure pretrained_stem is float32 (standard dtype for model weights)
        pretrained_stem = pretrained_stem.float()
        
        # ✅ Get the dtype from current model state to match
        target_dtype = current_stem.dtype

        if init_strategy == 'average':
            new_channel = pretrained_stem.mean(dim=1, keepdim=True)
        elif init_strategy == 'green':
            new_channel = pretrained_stem[:, 1:2, :, :]
        elif init_strategy == 'zero':
            new_channel = torch.zeros_like(pretrained_stem[:, 0:1, :, :])
        elif init_strategy == 'random':
            new_channel = torch.randn_like(pretrained_stem[:, 0:1, :, :]) * 0.02
        else:
            raise ValueError(f"Invalid init strategy: {init_strategy}, use average, green, zero, or random")

        # ✅ Ensure new_channel matches the target dtype
        new_channel = new_channel.to(dtype=target_dtype)
        
        # ✅ Ensure pretrained_stem matches before concatenation
        pretrained_stem = pretrained_stem.to(dtype=target_dtype)
        
        new_stem = torch.cat([pretrained_stem, new_channel], dim=1)
        
        # ✅ Ensure final tensor matches model's expected dtype
        model_state[stem_key] = new_stem.to(dtype=target_dtype)

        if stem_bias_key in pretrained:
            # ✅ Ensure bias dtype matches
            pretrained_bias = pretrained[stem_bias_key].to(dtype=target_dtype)
            model_state[stem_bias_key] = pretrained_bias
        
    else:
        raise ValueError(f"Stem key {stem_key} not found in pretrained weights\n\n\n{pretrained.keys()}")

    # ✅ Ensure all other weights match dtype
    for key in pretrained:
        if key != stem_key and key != stem_bias_key:
            if key in model_state and model_state[key].shape == pretrained[key].shape:
                # ✅ Preserve dtype consistency
                model_state[key] = pretrained[key].to(dtype=model_state[key].dtype)
            elif key in model_state:
                print(f"Warning: Key {key} found in model state but shape mismatch: model {model_state[key].shape} vs pretrained {pretrained[key].shape}")

    model.load_state_dict(model_state, strict=False)

    print(f"Loaded weights from {checkpoint_path}")
    print(f"Init strategy: {init_strategy}")
    return model


# -----------------------------
# Losses & metrics
# -----------------------------
class CSSReprojectionLoss(nn.Module):
    def __init__(self, wavelengths: np.ndarray, css: np.ndarray, tau: float = 10.0, p: float = 1.0):
        super().__init__()
        self.renderer = CSSRenderer(wavelengths, css)
        self.tau = tau
        self.p = p  # Lp norm; p=1 (L1) by default

    def forward(self, pred_hsi: torch.Tensor, gt_hsi: torch.Tensor, in_rgb_lin: torch.Tensor):
        # pred_hsi, gt_hsi: (B,Bands,H,W); in_rgb_lin: (B,C,H,W) where C can be 3 or 4
        l_hsi = F.l1_loss(pred_hsi, gt_hsi)
        
        # render predicted HSI back to RGB
        B, C, H, W = pred_hsi.shape  # C == bands
        Wcss = self.renderer.weight  # (C,3)

        # batched HSI(HW,C) @ (C,3) → RGB, for all items at once
        x = pred_hsi.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        rgb = x @ Wcss  # (B*H*W, 3)
        pred_rgb = rgb.view(B, H, W, 3).permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W)

        # Extract only RGB channels if input has 4 channels (RGB+NIR)
        in_rgb_only = in_rgb_lin[:, :3, :, :] if in_rgb_lin.shape[1] == 4 else in_rgb_lin
        
        l_rgb = F.l1_loss(pred_rgb, in_rgb_only)
        
        return l_hsi + self.tau * l_rgb, {'l_hsi': l_hsi.detach(), 'l_rgb': l_rgb.detach()}




# -----------------------------
# Training loop (minimal)
# -----------------------------
class AverageMeter:
    def __init__(self): self.v = 0; self.n = 0

    def add(self, x, k=1): self.v += float(x) * k; self.n += k

    @property
    def avg(self): return self.v / max(1, self.n)

def chk(name, t):
    if not torch.isfinite(t).all():
        raise RuntimeError(f"{name} has NaN/Inf: "
                           f"min={t.nanmin().item()} max={t.nanmax().item()}")

def train_one_epoch(model, loss_fn, opt, dl, device, scaler=None, log_every=50, is_dirty=False, check = True):
    model.train()
    # Initialize meters based on training mode
    if is_dirty:
        meters = {'loss': AverageMeter(), 'mrae': AverageMeter()}
    else:
        meters = {'loss': AverageMeter(), 'l_hsi': AverageMeter(), 'l_rgb': AverageMeter()}
    
    for it, batch in enumerate(dl):
        rgb = batch['rgb'].to(device).to(memory_format=torch.channels_last)  # (B,3,H,W)
        hsi = batch['hsi'].to(device).to(memory_format=torch.channels_last)  # (B,Bands,H,W)
        opt.zero_grad(set_to_none=True)
        if check:
            chk("rgb", rgb)
            chk("hsi", hsi)
        if scaler is None:
            pred = model(rgb)
            if check:
                chk("pred", pred)
            loss, parts = loss_fn(pred, hsi, rgb)
            if check:
                chk("loss", loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred = model(rgb)
                loss, parts = loss_fn(pred, hsi, rgb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        meters['loss'].add(loss.item(), rgb.size(0))
        # Update meters based on loss type
        for key in parts.keys():
            if key in meters:
                meters[key].add(parts[key].item(), rgb.size(0))
        
        if (it + 1) % log_every == 0:
            if is_dirty:
                print(f"it {it + 1}: loss={meters['loss'].avg:.4f} (mrae={meters['mrae'].avg:.4f})")
            else:
                print(f"it {it + 1}: loss={meters['loss'].avg:.4f} (hsi={meters['l_hsi'].avg:.4f}, rgb={meters['l_rgb'].avg:.4f})")
    return {k: m.avg for k, m in meters.items()}


@torch.no_grad()
def make_fg_mask(gt: torch.Tensor, threshold: float = 0.02, mode: str = "mean") -> torch.Tensor:
    """
    gt: (B, C, H, W) reflectance (≈ [0,1])
    Returns mask of shape (B, 1, H, W) with 1 = foreground.
    mode: "mean" uses mean over bands; "max" uses max over bands.
    """
    if mode == "max":
        stat = gt.abs().max(dim=1, keepdim=True).values
    else:  # "mean"
        stat = gt.abs().mean(dim=1, keepdim=True)
    return (stat > threshold).float()

@torch.no_grad()
def rmse(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(F.mse_loss(pred, gt))

@torch.no_grad()
def mrae(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Mean Relative Absolute Error over spectral channels
    return torch.mean(torch.abs(pred - gt) / (torch.abs(gt) + eps))

@torch.no_grad()
def rmse_masked(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # pred, gt: (B, C, H, W); mask: (B, 1, H, W)
    m = mask.expand_as(gt)
    num = m.sum().clamp(min=1)
    mse = (pred - gt) ** 2
    masked = torch.where(m.bool(), mse, torch.zeros_like(mse))
    mse = masked.sum() / num
    return torch.sqrt(mse)

@torch.no_grad()
def mrae_masked(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, eps: float = 0.02) -> torch.Tensor:
    # eps tied to reflectance scale to avoid tiny denominators (0.02 ~ 2% reflectance)
    m = mask.expand_as(gt)
    rae = (pred - gt).abs() / (gt.abs().clamp_min(eps))
    masked = torch.where(m.bool(), rae, torch.zeros_like(rae))
    num = m.sum().clamp(min=1)
    return masked.sum() / num

@torch.no_grad()
def evaluate(model, dl, device, mask_threshold: float = 0.02, mask_mode: str = "mean"):
    model.eval()
    rm_sum, rm_n = 0.0, 0
    mr_sum, mr_n = 0.0, 0
    for batch in dl:
        rgb = batch['rgb'].to(device)
        hsi = batch['hsi'].to(device)
        pred = model(rgb)

        mask = make_fg_mask(hsi, threshold=mask_threshold, mode=mask_mode)  # (B,1,H,W)

        rm = rmse_masked(pred, hsi, mask).item()
        mr = mrae_masked(pred, hsi, mask, eps=mask_threshold).item()

        bsz = rgb.size(0)
        rm_sum += rm * bsz;
        rm_n += bsz
        mr_sum += mr * bsz;
        mr_n += bsz

    return {'rmse': rm_sum / max(1, rm_n), 'mrae': mr_sum / max(1, mr_n)}

@torch.no_grad()
def evaluate_test(model, dl, dl_f, device, mask_threshold: float = 0.02, mask_mode: str = "mean"):
    model.eval()
    rm_sum, rm_n = 0.0, 0
    mr_sum, mr_n = 0.0, 0
    for batch, batch_f in dl, dl_f:
        rgb = batch['rgb'].to(device)
        hsi = batch['hsi'].to(device)

        rgb_f = batch_f['rgb'].to(device)

        pred = model(rgb)
        pred_f = model(rgb_f)

        pred_f = torch.rot90(pred_f, 2, dims=[1, 2])

        pred = (pred + pred_f) / 2 # yeah, peak of ensemble industry

        mask = make_fg_mask(hsi, threshold=mask_threshold, mode=mask_mode)  # (B,1,H,W)

        rm = rmse_masked(pred, hsi, mask).item()
        mr = mrae_masked(pred, hsi, mask, eps=mask_threshold).item()

        bsz = rgb.size(0)
        rm_sum += rm * bsz;
        rm_n += bsz
        mr_sum += mr * bsz;
        mr_n += bsz

    return {'rmse': rm_sum / max(1, rm_n), 'mrae': mr_sum / max(1, mr_n)}

class DirtyTrainingLoss(nn.Module):
    """Loss function for dirty training mode (render=False) with masked MRAE.
    Uses Mean Relative Absolute Error (MRAE) over foreground pixels only, similar to evaluate.
    Masks out background pixels using make_fg_mask with threshold.
    """

    def __init__(self, mask_threshold: float = 0.02, mask_mode: str = "mean", eps: float = 0.02):
        super().__init__()
        self.mask_threshold = mask_threshold
        self.mask_mode = mask_mode
        self.eps = eps  # eps tied to reflectance scale (0.02 ~ 2% reflectance)

    def forward(self, pred_hsi: torch.Tensor, gt_hsi: torch.Tensor, in_rgb_lin: torch.Tensor):
        # pred_hsi, gt_hsi: (B,Bands,H,W); in_rgb_lin: (B,3,H,W) - unused but kept for interface consistency
        # Create foreground mask from ground truth
        mask = make_fg_mask(gt_hsi, threshold=self.mask_threshold, mode=self.mask_mode)  # (B, 1, H, W)

        # Expand mask to match pred/gt shape
        m = mask.expand_as(gt_hsi)  # (B, Bands, H, W)

        # Compute relative absolute error
        rae = (pred_hsi - gt_hsi).abs() / (gt_hsi.abs().clamp_min(self.eps))  # (B, Bands, H, W)

        # Apply mask and compute mean over masked pixels only
        num = m.sum().clamp(min=1)
        loss = (rae * m).sum() / num

        return loss, {'mrae': loss.detach()}


def main():
    args = _NS(
        train_path='train_list.txt',
        val_path='valid_list.txt',
        test_path='test_list.txt',
        catalog_path='C:/Users/artem.chsherbakov/Documents/applejam/Catalogs/',  # Will be set on target machine
        cut_out = False,
        waves='fx10_wavelengths_204.npy',
        css='css_fx10_204.npy',  # Still needed for dataset renderer even if not used in loss
        device='cuda:0',
        epochs=150,
        bs=8,
        lr=3e-4,
        patch=240,
        render=True,  # Set to False for dirty training
        center_crop=False,
        channels=600,
        blocks=6,
        no_psnl=True, # need a beefy ass machine for that
        tau=10.0,  # Only used when render=True
        workers=16,
        val_frac=0.05,
        out='',  # Will be auto-generated
    )
    
    # Generate output filename
    out_string = 'awan_' + ('clean' if args.render else 'dirty')
    out_string += f'_center_crop' if args.center_crop else '_random_crop'
    out_string += f'_{args.patch}p' if args.patch else '_fp'
    out_string += f'_{args.channels}ch' + f'_{args.blocks}b'
    out_string += f'_nopsnl' if args.no_psnl else '_psnl'
    if args.render:
        out_string += f'_{args.tau}t'
    args.out = out_string

    def extract_paths(filename):
        paths = []
        with open(filename, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                dat = args.catalog_path + line.strip()
                hdr = args.catalog_path + line.split('.dat')[0] + '.hdr'
                id = line.split('/results/')[0].split('/')[-1]
                png = args.catalog_path + line.split('/results/')[0] + '/' + id + '.png'
                paths.append((hdr, dat, png))
        return paths

    device = torch.device(args.device)

    # Load file list
    train_pairs = extract_paths(args.train_path)
    test_pairs = extract_paths(args.test_path)
    val_pairs = extract_paths(args.val_path)

    if args.cut_out:
        train_pairs = train_pairs[:5]
        test_pairs = test_pairs[:1]
        val_pairs = val_pairs[:1]

    random.shuffle(train_pairs)

    print(len(train_pairs))
    print(len(val_pairs))
    waves = np.load(args.waves).astype('float32')   # (B,)
    css = np.load(args.css).astype('float32')       # (B,3)
    print('done load')

    # Create directories for logs and checkpoints
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    train_ds = UniversalReconstructionDataset(train_pairs, waves, css, patch_size=args.patch, augment=True, render=args.render, center_crop=args.center_crop)
    val_ds   = UniversalReconstructionDataset(val_pairs, waves, css, patch_size=None, augment=False, render=args.render)
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
    model = AWANCleanNet(bands_out=bands_out, channels=args.channels, n_blocks=args.blocks, use_psnl=not args.no_psnl).to(device)
    #model = torch.compile(model)  # PyTorch 2.x (optional but nice)
    #model.to(memory_format=torch.channels_last)
    
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
    
    # Document training bottlenecks for dirty training mode
    if not args.render:
        print("\n" + "="*60)
        print("DIRTY TRAINING MODE - BOTTLENECK ANALYSIS")
        print("="*60)
        print("When render=False, RGB images are loaded from PNG files instead of being rendered.")
        print("Potential bottlenecks:")
        print("1. I/O BOTTLENECK: PNG loading may be slower than CSS rendering for large datasets.")
        print("   - Each PNG file must be opened, decoded, and converted to linear RGB.")
        print("   - Consider caching RGB images in main memory if memory allows.")
        print("   - HSI data must still be loaded from ENVI files as before.")
        print("2. MEMORY BOTTLENECK: Original RGBs may have different memory footprint than rendered.")
        print("   - PNG images are typically 8-bit, but converted to float32 for training.")
        print("   - Consider using RGB caching strategy: cache only RGB in main memory, load HSI on-demand.")
        print("3. DATA LOADER BOTTLENECK: DataLoader workers may be underutilized if I/O bound.")
        print("   - Monitor worker utilization and adjust num_workers accordingly.")
        print("   - Consider using faster image loading libraries or preprocessing RGBs to disk cache.")
        print("="*60 + "\n")
    
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
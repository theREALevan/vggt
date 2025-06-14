from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import List, Tuple
import warnings
import sys
from contextlib import redirect_stdout
from io import StringIO

import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from tqdm import tqdm
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# Configure PIL to handle large images
Image.MAX_IMAGE_PIXELS = None  # Disable decompression bomb protection

class SuppressWarnings:
    """Context manager to suppress specific print statements."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._buffer = StringIO()
        sys.stdout = self._buffer
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        output = self._buffer.getvalue()
        sys.stdout = self._original_stdout
        # Only print lines that don't contain the warning
        for line in output.split('\n'):
            if line and not line.startswith('Warning: Found images with different shapes'):
                print(line)

################################################################################
# Geometry helpers
################################################################################

def normalize(vec: torch.Tensor) -> torch.Tensor:
    """L2-normalise the last dimension."""
    return vec / (vec.norm(dim=-1, keepdim=True).clamp(min=1e-8))


def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternion to rotation matrix.
    Args:
        quat: (..., 4) tensor in (w,x,y,z) order
    Returns:
        (..., 3, 3) rotation matrix tensor
    """
    q = normalize(quat)
    w, x, y, z = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w

    m00 = 1 - 2 * (yy + zz)
    m01 = 2 * (xy - zw)
    m02 = 2 * (xz + yw)
    m10 = 2 * (xy + zw)
    m11 = 1 - 2 * (xx + zz)
    m12 = 2 * (yz - xw)
    m20 = 2 * (xz - yw)
    m21 = 2 * (yz + xw)
    m22 = 1 - 2 * (xx + yy)

    row0 = torch.stack([m00, m01, m02], dim=-1)
    row1 = torch.stack([m10, m11, m12], dim=-1)
    row2 = torch.stack([m20, m21, m22], dim=-1)
    rot = torch.stack([row0, row1, row2], dim=-2)  # (..., 3, 3)
    return rot


def geodesic_loss(R_pred_rel: torch.Tensor, R_gt_rel: torch.Tensor) -> torch.Tensor:
    """Average SO(3) geodesic distance in radians."""
    m = torch.bmm(R_pred_rel, R_gt_rel.transpose(1, 2))
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    cos = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos).mean()

################################################################################
# Dataset & dataloader
################################################################################


class PairDataset(Dataset):
    """Dataset for Wiki/scene image pairs stored in *.npy dicts."""

    def __init__(self, npy_file: str) -> None:
        super().__init__()
        self.data: dict[int, dict] = np.load(npy_file, allow_pickle=True).item()
        self.keys: List[int] = list(self.data.keys())

    def __len__(self) -> int:
        return len(self.keys)

    def _build(self, img_info: dict) -> Tuple[str, torch.Tensor]:
        """Return image absolute-path string & quaternion tensor."""
        path = img_info["path"]

        quat = torch.tensor([
            img_info["qw"],
            img_info["qx"],
            img_info["qy"],
            img_info["qz"],
        ], dtype=torch.float32)
        return path, quat

    def __getitem__(self, idx: int):
        item = self.data[self.keys[idx]]
        p1, q1 = self._build(item["img1"])
        p2, q2 = self._build(item["img2"])
        overlap = item["overlap_amount"].lower()
        return (p1, p2, q1, q2, overlap)


# Collate function uses load_and_preprocess_images to batch-load & transform.

def collate_fn(batch):
    """
    Collate function to batch image pairs from different scenes.
    
    Args:
        batch: List of tuples from PairDataset.__getitem__()
               [(path1, path2, quat1, quat2, overlap), ...] length B
    
    Returns:
        images: (B, 2, 3, H, W) - B scenes, 2 images per scene, RGB channels, height, width
        q1: (B, 4) - quaternions for first image in each scene (w,x,y,z)
        q2: (B, 4) - quaternions for second image in each scene (w,x,y,z)
        overlaps: List[str] - overlap categories for each scene
        paths: List[Tuple[str, str]] - List of (path1, path2) tuples for each scene
    """
    
    # Unzip the batch into separate lists
    paths1, paths2, q1_list, q2_list, overlaps = zip(*batch)

    # Preprocess images to tensor (2B, 3, H, W) - suppress shape warnings
    with SuppressWarnings():
        # Load all images: paths1 + paths2 = [scene0_img1, scene1_img1, ..., scene0_img2, scene1_img2, ...]
        # Result: (2B, 3, H, W) where 2B = total images from B scenes
        imgs_all = load_and_preprocess_images(list(paths1) + list(paths2))
    B = len(batch)
    
    # Reshape to group image pairs by scene:
    # (2B, 3, H, W) -> (2, B, 3, H, W) -> (B, 2, 3, H, W)
    # Groups [scene0_img1, scene1_img1, ...] and [scene0_img2, scene1_img2, ... ]
    # into [(scene0_img1, scene0_img2), (scene1_img1, scene1_img2), ...]
    imgs_all = imgs_all.view(2, B, *imgs_all.shape[1:]).permute(1, 0, 2, 3, 4)

    # Stack quaternions: List[Tensor(4,)] -> Tensor(B, 4)
    q1 = torch.stack(q1_list)  # (B, 4) - first image quaternions for B scenes
    q2 = torch.stack(q2_list)  # (B, 4) - second image quaternions for B scenes
    
    # Create list of path pairs
    paths = list(zip(paths1, paths2))
    
    return imgs_all, q1, q2, overlaps, paths

################################################################################
# Training utilities
################################################################################


def create_loader(overlap_npy: str, none_npy: str, batch_size: int,
                  num_workers: int, pin_memory: bool, upweight_none: float = 1.0):
    overlap_ds = PairDataset(overlap_npy)
    none_ds = PairDataset(none_npy)

    combined = ConcatDataset([overlap_ds, none_ds])

    # Weight sampling to balance classes
    weights = ([1.0 / len(overlap_ds)] * len(overlap_ds) +
               [upweight_none / len(none_ds)] * len(none_ds))
    sampler = WeightedRandomSampler(weights, num_samples=len(combined), replacement=True)

    loader = DataLoader(
        combined,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    return loader, len(overlap_ds), len(none_ds)

################################################################################
# Main training loop
################################################################################


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (
        torch.bfloat16
        if (device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8)
        else torch.float16
    )

    # Dataloader
    loader, n_overlap, n_none = create_loader(
        args.overlap_npy,
        args.none_npy,
        args.batch_size,
        args.num_workers,
        pin_memory=(device.type == "cuda"),
        upweight_none=args.upweight_none,
    )

    print(f"Loaded {n_overlap} overlap and {n_none} none-overlap samples.")

    # Model
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    if args.resume_ckpt and Path(args.resume_ckpt).is_file():
        print(f"Loading checkpoint {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
    model.train()

    # Optimiser & scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler()

    # Training epochs with tqdm logging
    outer_bar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch")
    for epoch in outer_bar:
        running_loss = 0.0
        epoch_bar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch", leave=False)

        for images, q1_gt, q2_gt, overlaps, paths in epoch_bar:
            """
            Training step for one batch of image pairs from different scenes.
            
            Input shapes:
                images: (B, 2, 3, H, W) - B scenes, 2 images per scene, RGB, height, width
                q1_gt: (B, 4) - ground truth quaternions for first image in each scene
                q2_gt: (B, 4) - ground truth quaternions for second image in each scene
                overlaps: List[str] - overlap categories for each scene
                paths: List[Tuple[str, str]] - List of (path1, path2) tuples for each scene
            """
            images = images.to(device, non_blocking=True)
            q1_gt = q1_gt.to(device)
            q2_gt = q2_gt.to(device)

            B = images.size(0)  # Number of scenes in this batch
            
            # Process each pair independently
            batch_losses = []
            for i in range(B):
                # Extract one scene's image pair
                pair_imgs = images[i]  # (2, 3, H, W) - 2 images from scene i, RGB, height, width
                
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    # VGGT forward pass for this scene's image pair
                    preds = model(pair_imgs)  # Input: (2, 3, H, W) -> Output: dict with 'pose_enc'
                    
                    # Check for NaN/Inf in predictions
                    if torch.isnan(preds["pose_enc"]).any() or torch.isinf(preds["pose_enc"]).any():
                        path1, path2 = paths[i]
                        print(f"\nNaN/Inf detected in pose_enc for image pair:")
                        print(f"  Image 1: {path1}")
                        print(f"  Image 2: {path2}")
                        print(f"  Overlap type: {overlaps[i]}")
                        print(f"  GT quaternion 1: {q1_gt[i].cpu().numpy()}")
                        print(f"  GT quaternion 2: {q2_gt[i].cpu().numpy()}")
                        raise ValueError("NaN/Inf detected in pose_enc")
                    
                    # Get image dimensions for pose decoding
                    H, W = pair_imgs.shape[-2:]
                    
                    # Decode pose encodings to extrinsic matrices
                    # preds["pose_enc"]: (1, 2, 9) - 1 batch, 2 images, 9-dim pose encoding
                    extr, _ = pose_encoding_to_extri_intri(preds["pose_enc"], (H, W))
                    # extr: (1, 2, 3, 4) - 1 batch, 2 cameras, 3x4 extrinsic matrices [R|t]
                    
                    # Check for NaN/Inf in extrinsic matrices
                    if torch.isnan(extr).any() or torch.isinf(extr).any():
                        path1, path2 = paths[i]
                        print(f"\nNaN/Inf detected in extrinsic matrices for image pair:")
                        print(f"  Image 1: {path1}")
                        print(f"  Image 2: {path2}")
                        print(f"  Overlap type: {overlaps[i]}")
                        print(f"  GT quaternion 1: {q1_gt[i].cpu().numpy()}")
                        print(f"  GT quaternion 2: {q2_gt[i].cpu().numpy()}")
                        raise ValueError("NaN/Inf detected in extrinsic matrices")
                    
                    # Remove batch dimension since we process one scene at a time
                    extr = extr.squeeze(0)  # (1, 2, 3, 4) -> (2, 3, 4) - 2 cameras, 3x4 matrices
                    
                    # Extract 3x3 rotation matrices from 3x4 extrinsic matrices
                    R_pred_pair = extr[:, :3, :3]  # (2, 3, 4) -> (2, 3, 3) - 2 rotation matrices
                    
                    # Convert ground truth quaternions to rotation matrices for this scene
                    R1_gt_i = quat_to_rotmat(q1_gt[i:i+1])  # (1, 4) -> (1, 3, 3) - first image rotation
                    R2_gt_i = quat_to_rotmat(q2_gt[i:i+1])  # (1, 4) -> (1, 3, 3) - second image rotation
                    
                    # Compute relative rotations: R_rel = R2 @ R1^T
                    R1_pred = R_pred_pair[0]  # (3, 3) - first image predicted rotation
                    R2_pred = R_pred_pair[1]  # (3, 3) - second image predicted rotation
                    # Relative rotation: how to rotate from camera 1 to camera 2
                    R_pred_rel = torch.matmul(R2_pred, R1_pred.transpose(-2, -1)).unsqueeze(0)  # (3,3) -> (1,3,3)
                    R_gt_rel = torch.matmul(R2_gt_i, R1_gt_i.transpose(1, 2))  # (1,3,3) @ (1,3,3) -> (1,3,3)
                    
                    # Check for NaN/Inf in relative rotations
                    if torch.isnan(R_pred_rel).any() or torch.isinf(R_pred_rel).any():
                        path1, path2 = paths[i]
                        print(f"\nNaN/Inf detected in relative rotations for image pair:")
                        print(f"  Image 1: {path1}")
                        print(f"  Image 2: {path2}")
                        print(f"  Overlap type: {overlaps[i]}")
                        print(f"  GT quaternion 1: {q1_gt[i].cpu().numpy()}")
                        print(f"  GT quaternion 2: {q2_gt[i].cpu().numpy()}")
                        raise ValueError("NaN/Inf detected in relative rotations")
                    
                    # Compute geodesic distance between predicted and ground truth relative rotations
                    pair_loss = geodesic_loss(R_pred_rel, R_gt_rel)
                    
                    # Check for NaN/Inf in loss
                    if torch.isnan(pair_loss) or torch.isinf(pair_loss):
                        path1, path2 = paths[i]
                        print(f"\nNaN/Inf detected in loss computation for image pair:")
                        print(f"  Image 1: {path1}")
                        print(f"  Image 2: {path2}")
                        print(f"  Overlap type: {overlaps[i]}")
                        print(f"  GT quaternion 1: {q1_gt[i].cpu().numpy()}")
                        print(f"  GT quaternion 2: {q2_gt[i].cpu().numpy()}")
                        raise ValueError("NaN/Inf detected in loss computation")
                    
                    batch_losses.append(pair_loss)
            
            # Average loss across all scenes in the batch
            loss = torch.stack(batch_losses).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * B
            # Update progress bar with current batch loss and running average
            current_avg_loss = running_loss / ((epoch_bar.n + 1) * args.batch_size)
            epoch_bar.set_postfix({
                'batch_loss': f'{loss.item():.3f}',
                'avg_loss': f'{current_avg_loss:.3f}'
            })

        epoch_bar.close()

        mean_loss = running_loss / (len(loader) * args.batch_size)
        outer_bar.set_postfix(mean_loss=f"{mean_loss:.3f} rad / {(mean_loss*180/np.pi):.2f} deg")

        # Checkpoint
        if args.out_ckpt:
            ckpt_path = Path(args.out_ckpt) / f"vggt_epoch_{epoch}.pth"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict()}, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")


################################################################################
# CLI
################################################################################


def parse_args():
    p = argparse.ArgumentParser(description="Train VGGT with custom geodesic loss on pair datasets.")
    p.add_argument("--overlap-npy", type=str, default="metadata/train_overlap_megascenes_path_valid.npy")
    p.add_argument("--none-npy", type=str, default="metadata/train_none_megascenes_path_valid.npy")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--upweight-none", type=float, default=1.0, help="Sampling weight multiplier for 'none' overlap class.")
    p.add_argument("--out-ckpt", type=str, default="checkpoints", help="Directory to save checkpoints (set empty to disable).")
    p.add_argument("--resume-ckpt", type=str, default="", help="Path to checkpoint to resume from.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args()) 
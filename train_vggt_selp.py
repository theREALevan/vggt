import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from vggt.models.vggt import VGGT
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.load_fn import load_and_preprocess_images
import json
import torch.multiprocessing as mp
from tqdm import tqdm

mp.set_start_method('spawn', force=True)

def compute_geodesic_distance_from_two_matrices(m1, m2):
    """Compute the geodesic distance between two rotation matrices."""
    batch = m1.shape[0]
    m = torch.bmm(m1.to(torch.float64), m2.to(torch.float64).transpose(1, 2))  # batch*3*3

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda()) * -1)

    theta = torch.acos(cos)

    return theta

def normalize_vector(v):
    """Normalize a batch of vectors."""
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, torch.tensor([1e-8], device=v.device))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v/v_mag
    return v

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix."""
    # Stack quaternion components and normalize
    quaternion = torch.stack([qw, qx, qy, qz], dim=0).unsqueeze(0)  # Add batch dimension
    batch = quaternion.shape[0]
    
    quat = normalize_vector(quaternion)
    
    qw = quat[..., 0].view(batch, 1)
    qx = quat[..., 1].view(batch, 1)
    qy = quat[..., 2].view(batch, 1)
    qz = quat[..., 3].view(batch, 1)
    
    # Unit quaternion rotation matrices computation
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw
    
    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3
    
    return matrix.squeeze(0)

def compute_rotation_loss(pose_enc, gt_rotation, image_size_hw):
    """
    Compute the geodesic loss between predicted rotation and ground truth rotation.
    
    Args:
        pose_enc: Tensor of shape [B, 2, 9] - predicted pose encoding
        gt_rotation: Tensor of shape [B, 2, 3, 3] - ground truth rotation matrices for both images
        image_size_hw: Tuple (H, W) - image dimensions
    
    Returns:
        loss: Scalar tensor of geodesic error
    """
    # Convert pose encoding to rotation matrices
    extrinsic_t, _ = pose_encoding_to_extri_intri(pose_enc, image_size_hw=image_size_hw)
    pred_R1 = extrinsic_t[:, 0, :3, :3]  # First image rotation
    pred_R2 = extrinsic_t[:, 1, :3, :3]  # Second image rotation
    
    # Extract ground truth rotations
    gt_R1 = gt_rotation[:, 0]  # First image rotation
    gt_R2 = gt_rotation[:, 1]  # Second image rotation
    
    # Compute relative rotations
    pred_rel_R = torch.bmm(pred_R2, pred_R1.transpose(1, 2))
    gt_rel_R = torch.bmm(gt_R2, gt_R1.transpose(1, 2))
    
    # Compute geodesic error
    error = compute_geodesic_distance_from_two_matrices(pred_rel_R, gt_rel_R)
    
    return error.mean()

class MegaScenesDataset(Dataset):
    def __init__(self, data_path, base_paths=None, categories_json='categories.json', mode="pad"):
        self.data = np.load(data_path, allow_pickle=True).item()
        self.mode = mode  # "crop" or "pad"
        # List of base paths to search
        if base_paths is None:
            self.base_paths = [
                '/share/phoenix/nfs05/S8/jt664/WikiSFM/data/main',
                '/share/phoenix/nfs06/S9/jt664/megascenes_local/data'
            ]
        else:
            self.base_paths = base_paths
        self.keys = list(self.data.keys())
        with open(categories_json, 'r') as f:
            cat_map = json.load(f)
        self.sceneid_to_category = {int(v): k for k, v in cat_map.items()}
        
        # Pre-compute valid paths and store minimal data
        self.valid_pairs = []
        # Debug counters
        total_pairs = len(self.keys)
        invalid_path_format = 0
        missing_category = 0
        missing_images = 0
        
        for idx, key in enumerate(self.keys):
            pair_data = self.data[key]
            img1_path = self.get_img_path(pair_data['img1']['path'])
            img2_path = self.get_img_path(pair_data['img2']['path'])
            
            # Track why pairs are being filtered
            if img1_path is None or img2_path is None:
                if not pair_data['img1']['path'].startswith('images/') or not pair_data['img2']['path'].startswith('images/'):
                    invalid_path_format += 1
                else:
                    parts1 = pair_data['img1']['path'].split('/')
                    parts2 = pair_data['img2']['path'].split('/')
                    scene_id1 = int(parts1[1] + parts1[2])
                    scene_id2 = int(parts2[1] + parts2[2])
                    if self.sceneid_to_category.get(scene_id1) is None or self.sceneid_to_category.get(scene_id2) is None:
                        missing_category += 1
                    else:
                        missing_images += 1
                continue
                
            # Store only necessary data
            self.valid_pairs.append({
                'idx': idx,
                'img1_path': img1_path,
                'img2_path': img2_path,
                'overlap_amount': pair_data['overlap_amount'],
                'img1_quat': [pair_data['img1']['qw'], pair_data['img1']['qx'], 
                            pair_data['img1']['qy'], pair_data['img1']['qz']],
                'img2_quat': [pair_data['img2']['qw'], pair_data['img2']['qx'], 
                            pair_data['img2']['qy'], pair_data['img2']['qz']]
            })
            
        print(f"\nDataset filtering statistics:")
        print(f"Total pairs in file: {total_pairs}")
        print(f"Valid pairs found: {len(self.valid_pairs)}")
        print(f"Pairs filtered out: {total_pairs - len(self.valid_pairs)}")
        print(f"  - Invalid path format: {invalid_path_format}")
        print(f"  - Missing category mapping: {missing_category}")
        print(f"  - Missing image files: {missing_images}")
        
        # Clear the original data to free memory
        del self.data
        import gc
        gc.collect()

    def __len__(self):
        return len(self.valid_pairs)

    def get_img_path(self, img_path):
        parts = img_path.split('/')
        if parts[0] == 'images':
            scene_id = int(parts[1] + parts[2])
            category = self.sceneid_to_category.get(scene_id)
            if category is None:
                return None
            parts = [parts[0], category] + parts[3:]
            for base in self.base_paths:
                candidate = os.path.join(base, *parts)
                if os.path.exists(candidate):
                    return candidate
            return None
        else:
            return None

    def __getitem__(self, idx):
        pair_data = self.valid_pairs[idx]
        
        # Load and preprocess images sequentially
        img1 = load_and_preprocess_images([pair_data['img1_path']], mode=self.mode)[0]  # [3, H, W]
        img2 = load_and_preprocess_images([pair_data['img2_path']], mode=self.mode)[0]  # [3, H, W]
        
        # Convert quaternions to rotation matrices
        gt_R1 = quaternion_to_rotation_matrix(
            torch.tensor(pair_data['img1_quat'][0]),
            torch.tensor(pair_data['img1_quat'][1]),
            torch.tensor(pair_data['img1_quat'][2]),
            torch.tensor(pair_data['img1_quat'][3])
        )
        
        gt_R2 = quaternion_to_rotation_matrix(
            torch.tensor(pair_data['img2_quat'][0]),
            torch.tensor(pair_data['img2_quat'][1]),
            torch.tensor(pair_data['img2_quat'][2]),
            torch.tensor(pair_data['img2_quat'][3])
        )
        
        # Stack rotations
        gt_rotation = torch.stack([gt_R1, gt_R2], dim=0)
        
        return {
            'img1': img1,  # [3, H, W]
            'img2': img2,  # [3, H, W]
            'rotation': gt_rotation,  # [2, 3, 3]
            'overlap_amount': pair_data['overlap_amount']
        }

def custom_collate_fn(batch):
    # Extract and stack images
    img1_list = [item['img1'] for item in batch]
    img2_list = [item['img2'] for item in batch]
    rotations = [item['rotation'] for item in batch]
    overlap_amounts = [item['overlap_amount'] for item in batch]
    
    # Stack images into tensors
    img1_batch = torch.stack(img1_list)  # [B, 3, H, W]
    img2_batch = torch.stack(img2_list)  # [B, 3, H, W]
    images = torch.stack([img1_batch, img2_batch], dim=1)  # [B, 2, 3, H, W]
    
    # Stack rotations
    rotations = torch.stack(rotations)  # [B, 2, 3, 3]
    
    # Clear lists to free memory
    del img1_list, img2_list
    import gc
    gc.collect()
    
    return {
        'images': images,  # [B, 2, 3, H, W] tensor
        'rotation': rotations,  # [B, 2, 3, 3] tensor
        'overlap_amount': overlap_amounts
    }

def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        max_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"\nGPU Memory Usage:")
        print(f"Allocated: {allocated:.1f}MB")
        print(f"Reserved:  {reserved:.1f}MB")
        print(f"Max Memory: {max_memory:.1f}MB")
        print(f"Free:      {max_memory - allocated:.1f}MB")

def train_epoch(model, dataloader, optimizer, scaler, device, accumulation_steps=4):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    print(f"\nStarting epoch with {len(dataloader)} batches")
    print(f"Batch size: {dataloader.batch_size}, Accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {dataloader.batch_size * accumulation_steps}")
    print_gpu_memory()

    pbar = tqdm(dataloader, desc="Training", leave=True)
    
    for batch_idx, batch in enumerate(pbar):
        batch_loss = None
        try:
            # Print memory before data loading
            # print(f"\nData loading memory - Batch {batch_idx}:")
            # print_gpu_memory()
            
            # Clear cache before processing each batch
            torch.cuda.empty_cache()
            
            images = batch['images'].to(device)  # [B, 2, 3, H, W]
            gt_rotation = batch['rotation'].to(device)  # [B, 2, 3, 3]
            H, W = images.shape[-2:]
            
            # Forward pass with mixed precision
            with autocast():
                # Get aggregated tokens
                aggregated_tokens_list, ps_idx = model.aggregator(images)
                
                # Predict camera poses
                pose_enc = model.camera_head(aggregated_tokens_list)[-1]  # [B, 2, 9]
                
                # Compute geodesic loss
                batch_loss = compute_rotation_loss(pose_enc, gt_rotation, image_size_hw=(H, W))
                batch_loss = batch_loss / accumulation_steps  # Normalize loss for gradient accumulation
            
            # Backward pass with gradient scaling
            scaler.scale(batch_loss).backward()
            
            # Print training memory after forward/backward
            # print(f"Training memory - Batch {batch_idx}:")
            # print_gpu_memory()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # Clear cache after optimizer step
                torch.cuda.empty_cache()
            
            # Clear memory
            del images, gt_rotation, aggregated_tokens_list, ps_idx, pose_enc
            torch.cuda.empty_cache()
            
            if batch_loss is not None:  # Only update total_loss if batch_loss was computed
                total_loss += batch_loss.item() * accumulation_steps  # Scale loss back up for reporting
                
                pbar.set_postfix({
                    'loss': f'{batch_loss.item() * accumulation_steps:.4f}',
                    'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
                    'memory': f'{torch.cuda.memory_allocated() / 1024**2:.1f}MB'
                })
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                print(f"\nWARNING: out of memory in batch {batch_idx}. Skipping batch.")
                print_gpu_memory()
                continue
            else:
                raise e
    
    return total_loss / len(dataloader)

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    num_epochs = 50
    learning_rate = 1e-4
    accumulation_steps = 8
    num_workers = 2
    prefetch_factor = 2
    pin_memory = True
    
    print("\nInitializing training...")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {batch_size * accumulation_steps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Number of workers: {num_workers}")
    print(f"Prefetch factor: {prefetch_factor}")
    print(f"Pin memory: {pin_memory}")
    
    # Initialize model
    print("\nLoading VGGT model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    print("Model loaded successfully")
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Enable gradient checkpointing for transformer blocks
    print("\nEnabling gradient checkpointing...")
    checkpoint_count = 0
    for module in model.modules():
        if hasattr(module, 'use_checkpoint'):
            module.use_checkpoint = True
            module.use_reentrant = False
            checkpoint_count += 1
    print(f"Enabled checkpointing for {checkpoint_count} modules")
    
    # Enable memory efficient attention if available
    if hasattr(model, 'enable_memory_efficient_attention'):
        print("\nEnabling memory efficient attention...")
        model.enable_memory_efficient_attention()
        print("Memory efficient attention enabled")
    
    # Create datasets - combine both overlap and none-overlap data
    print("\nLoading datasets...")
    train_none = MegaScenesDataset('metadata/train_none_megascenes_path.npy', mode="pad")
    train_overlap = MegaScenesDataset('metadata/train_overlap_megascenes_path.npy', mode="pad")
    print(f"Loaded non-overlap dataset with {len(train_none)} samples")
    print(f"Loaded overlap dataset with {len(train_overlap)} samples")
    
    # Create data loaders with optimized settings
    train_none_loader = DataLoader(
        train_none, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        collate_fn=custom_collate_fn,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        drop_last=True,
        generator=torch.Generator().manual_seed(42)
    )
    
    train_overlap_loader = DataLoader(
        train_overlap,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        drop_last=True,
        generator=torch.Generator().manual_seed(43)
    )
    
    print(f"Created data loaders with {len(train_none_loader)} batches for non-overlap and {len(train_overlap_loader)} batches for overlap")
    
    # Initialize optimizer and scaler
    print("\nInitializing optimizer and scaler...")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    scaler = GradScaler()
    print("Optimizer and scaler initialized")
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*50}")
        
        # Alternate between overlap and non-overlap data
        if epoch % 2 == 0:
            print("\nTraining on non-overlap data...")
            none_loss = train_epoch(model, train_none_loader, optimizer, scaler, device, accumulation_steps)
            print(f"Non-overlap Loss: {none_loss:.4f}")
        else:
            print("\nTraining on overlap data...")
            overlap_loss = train_epoch(model, train_overlap_loader, optimizer, scaler, device, accumulation_steps)
            print(f"Overlap Loss: {overlap_loss:.4f}")
        
        print(f"\nEpoch {epoch+1}/{num_epochs} completed")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            print(f"\nSaving checkpoint for epoch {epoch+1}...")
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'none_loss': none_loss if epoch % 2 == 0 else overlap_loss,
            }
            torch.save(checkpoint, f'vggt_megascenes_checkpoint_epoch{epoch+1}.pth')
            print("Checkpoint saved successfully")
            
        # Clear cache after each epoch
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()

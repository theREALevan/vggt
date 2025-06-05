import os
import torch
import numpy as np
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from tqdm import tqdm

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
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v/v_mag
    return v

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix using the original implementation."""
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

def evaluate_pair(model, img1_path, img2_path, gt_R1, gt_R2, device, dtype):
    """Evaluate a pair of images."""
    # Load and preprocess both images in a batch
    images = load_and_preprocess_images([img1_path, img2_path]).to(device)
    H, W = images.shape[-2:]
    
    # Run inference on the batch
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    
    # Get predicted rotations
    extrinsic_t, _ = pose_encoding_to_extri_intri(
        predictions["pose_enc"],     # shape (1, S, 9)
        image_size_hw=(H, W)
    )
    extrinsic_t = extrinsic_t.squeeze(0)  # Remove batch dimension, (S, 3, 4)
    
    # Extract rotation matrices
    pred_R1 = extrinsic_t[0, :3, :3]
    pred_R2 = extrinsic_t[1, :3, :3]
    
    # Compute relative rotations
    pred_rel_R = torch.mm(pred_R2, pred_R1.transpose(0, 1))
    gt_rel_R = torch.mm(gt_R2, gt_R1.transpose(0, 1))
    
    # Compute geodesic error
    error = compute_geodesic_distance_from_two_matrices(
        pred_rel_R.unsqueeze(0), 
        gt_rel_R.unsqueeze(0)
    )
    
    return error.item() * 180 / np.pi

def print_metrics(errors, category):
    """Print metrics for a given category of errors."""
    print(f"\nResults for {category} overlap:")
    print(f"MGE: {np.median(errors):.2f}°")
    print(f"Mean error: {np.mean(errors):.2f}°")
    print(f"RRA@15°: {100 * np.mean(np.array(errors) <= 15):.2f}%")
    print(f"RRA@30°: {100 * np.mean(np.array(errors) <= 30):.2f}%")
    return {
        'mge': np.median(errors),
        'mean_error': np.mean(errors),
        'rra_15': np.mean(np.array(errors) <= 15),
        'rra_30': np.mean(np.array(errors) <= 30)
    }

def main():
    # Setup device & model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (
        device=="cuda" and torch.cuda.get_device_capability()[0] >= 8
    ) else torch.float16
    
    print(f"Device: {device}, AMP dtype: {dtype}")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    
    # Load sELP test set
    test_data = np.load('/home/yz864/vggt/metadata/selp_test_set.npy', allow_pickle=True).item()
    
    errors_by_overlap = {
        'large': [],
        'small': [],
        'none': []
    }
    
    # Process all pairs
    for idx in tqdm(test_data.keys()):
        pair_data = test_data[idx]
        overlap_amount = pair_data['overlap_amount'].lower()
        
        # Get image paths
        img1_path = os.path.join('/home/yz864/vggt/metadata/images_to_npys/test_scenes_images/selp', pair_data['img1']['path'])
        img2_path = os.path.join('/home/yz864/vggt/metadata/images_to_npys/test_scenes_images/selp', pair_data['img2']['path'])
        
        gt_R1 = quaternion_to_rotation_matrix(
            torch.tensor(pair_data['img1']['qw'], device=device),
            torch.tensor(pair_data['img1']['qx'], device=device),
            torch.tensor(pair_data['img1']['qy'], device=device),
            torch.tensor(pair_data['img1']['qz'], device=device)
        )
        
        gt_R2 = quaternion_to_rotation_matrix(
            torch.tensor(pair_data['img2']['qw'], device=device),
            torch.tensor(pair_data['img2']['qx'], device=device),
            torch.tensor(pair_data['img2']['qy'], device=device),
            torch.tensor(pair_data['img2']['qz'], device=device)
        )
        
        error = evaluate_pair(model, img1_path, img2_path, gt_R1, gt_R2, device, dtype)
        errors_by_overlap[overlap_amount].append(error)
        
        # Print progress every 100 pairs
        total_processed = sum(len(errors) for errors in errors_by_overlap.values())
        if total_processed % 100 == 0:
            print(f"\nProcessed {total_processed} pairs")
            for overlap in errors_by_overlap:
                if errors_by_overlap[overlap]:
                    print_metrics(errors_by_overlap[overlap], overlap)
    
    # Final results
    print("\n=== Final Results ===")
    results = {}
    
    # Create a text file for results
    with open('selp_test_results.txt', 'w') as f:
        f.write("VGGT Results on sELP Dataset\n")
        f.write("==========================\n\n")
        
        # Table header
        f.write(f"{'Overlap':10} {'MGE':>10} {'RRA15':>10} {'RRA30':>10}\n")
        f.write("-" * 45 + "\n")
        
        for overlap in ['large', 'small', 'none']:
            if errors_by_overlap[overlap]:
                metrics = print_metrics(errors_by_overlap[overlap], overlap)
                results[overlap] = metrics
                
                # Write to text file
                f.write(f"{overlap:10} {metrics['mge']:10.2f} {metrics['rra_15']*100:10.2f} {metrics['rra_30']*100:10.2f}\n")
        
        # Add total number of pairs
        f.write("\nTotal pairs processed:\n")
        for overlap in ['large', 'small', 'none']:
            f.write(f"{overlap}: {len(errors_by_overlap[overlap])} pairs\n")
    
    # Save detailed results as numpy file
    np.save('selp_test_results.npy', {
        'errors_by_overlap': errors_by_overlap,
        'results': results
    })
    
    print("\nResults have been saved to:")
    print("- selp_test_results.txt")
    print("- selp_test_results.npy")

if __name__ == '__main__':
    main() 
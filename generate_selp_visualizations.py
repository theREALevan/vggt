import os
import torch
import numpy as np
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shutil

def get_fov(size, fl):
    fov_x = np.degrees(2*np.arctan(size[0]/(2*fl[0])))
    fov_y = np.degrees(2*np.arctan(size[1]/(2*fl[1])))
    return fov_x, fov_y

def draw_ellipse(ax, fovx, fovy, tilt_angle, latitude, longitude, color):
    a = np.deg2rad(fovx)
    b = np.deg2rad(fovy)
    theta = np.linspace(0, 2 * np.pi, 100)
    x = a * np.cos(theta)
    y = b * np.sin(theta)

    # Rotate the ellipse based on tilt angle
    R = np.array([[np.cos(tilt_angle), -np.sin(tilt_angle)], [np.sin(tilt_angle), np.cos(tilt_angle)]])
    ellipse = np.dot(R, np.array([x, y]))

    # Plot the ellipse on the ellipsoid
    if color == 'cyan':
        ax.plot(longitude + ellipse[0], latitude + ellipse[1], color=color, linewidth=2, linestyle='dashed')
    else:
        ax.plot(longitude + ellipse[0], latitude + ellipse[1], color=color, linewidth=2)

def rotation_matrix_to_euler_angles(R):
    """Convert rotation matrix to Euler angles (roll, pitch, yaw) in degrees.
    Using the ZYX convention, accounting for OpenCV coordinate system.
    
    In OpenCV convention:
    - x-axis points right
    - y-axis points down
    - z-axis points forward
    
    The Euler angles represent:
    - Roll: rotation around the z-axis (forward axis)
    - Pitch: rotation around the x-axis (right axis)
    - Yaw: rotation around the y-axis (down axis)
    """
    # Extract Euler angles from rotation matrix
    # First get pitch (around x-axis)
    pitch = np.arctan2(-R[2, 1], R[2, 2])
    
    # Then get roll and yaw
    if abs(R[2, 2]) > 1e-6 or abs(R[2, 1]) > 1e-6:  # Not at pitch = ±90°
        roll = np.arctan2(R[1, 0], R[0, 0])
        yaw = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    else:
        # Gimbal lock case
        roll = 0
        yaw = np.arctan2(-R[2, 0], 0)
    
    # Convert to degrees and return in roll, pitch, yaw order
    return np.array([roll, pitch, yaw]) * 180 / np.pi

def save_ellipsoid(data_array, gt_angles, est_angles, id, out_dir):
    # Load images
    img1_path = os.path.join('metadata/images_to_npys/test_scenes_images/selp', data_array[id]['img1']['path'])
    img2_path = os.path.join('metadata/images_to_npys/test_scenes_images/selp', data_array[id]['img2']['path'])
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # Set up figure with 3 columns: image1 | image2 | ellipsoid
    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 2])

    # Show image 1 with red border
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_edgecolor('tomato')
        spine.set_linewidth(6)
        spine.set_visible(True)

    # Show image 2 with blue border
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_edgecolor('cornflowerblue')
        spine.set_linewidth(6)
        spine.set_visible(True)

    # Ellipsoid plot
    ax3 = fig.add_subplot(gs[0, 2], projection="mollweide")

    fl1 = (data_array[id]['img1']['fx'], data_array[id]['img1']['fy'])
    fl2 = (data_array[id]['img2']['fx'], data_array[id]['img2']['fy'])
    size1 = img1.size
    size2 = img2.size
    fovx1, fovy1 = get_fov(size1, fl1)
    fovx2, fovy2 = get_fov(size2, fl2)

    # Draw ellipses with correct rotation angles
    draw_ellipse(ax3, fovx1/2, fovy1/2, 0, 0, 0, 'tomato')
    draw_ellipse(ax3, fovx2/2, fovy2/2, 0, np.deg2rad(gt_angles[1]), -np.deg2rad(gt_angles[2]), 'cornflowerblue')
    draw_ellipse(ax3, fovx2/2, fovy2/2, 0, np.deg2rad(est_angles[1]), -np.deg2rad(est_angles[2]), 'cyan')

    y_offset = -30
    yticks = np.array([-60, -30, 0, 30, 60])
    yticks_minor = np.arange(-75, 90, 15)
    ax3.set_yticks(yticks_minor * np.pi / 180, minor=True)
    ax3.set_yticks(yticks * np.pi / 180, [f"{y}°" for y in yticks], fontsize=14)
    xticks = np.array([-90, 0, 90])
    xticks_minor = np.arange(-150, 180, 30)
    ax3.set_xticks(xticks * np.pi / 180, [])
    ax3.set_xticks(xticks_minor * np.pi / 180, minor=True)
    for xtick in xticks:
        x = xtick * np.pi / 180
        y = y_offset * np.pi / 180
        ax3.text(x, y, f"{xtick}°", ha="center", va="center", fontsize=14)
        ax3.grid(which="minor")
        ax3.grid(which="major")
    red_patch = mpatches.Patch(color='tomato', label='Origin-Image 1')
    blue_patch = mpatches.Patch(color='cornflowerblue', label='Relative Rotation-Image 2')
    green_patch = mpatches.Patch(color='cyan', label='Estimated Rotation-Image 2')
    ax3.legend(handles=[red_patch, blue_patch, green_patch], loc='upper right', fontsize='x-small')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{id}_ellipsoid.png"))
    plt.close()

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

def normalize_vector(v):
    """Normalize a batch of vectors."""
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v/v_mag
    return v

def main():
    # Setup device & model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (
        device=="cuda" and torch.cuda.get_device_capability()[0] >= 8
    ) else torch.float16
    
    print(f"Device: {device}, AMP dtype: {dtype}")
    
    # Load saved data
    print("Loading saved data...")
    viz_pairs = np.load('selp_viz_pairs.npy', allow_pickle=True).item()
    test_data = np.load('metadata/selp_test_set.npy', allow_pickle=True).item()
    
    # Create output directory for visualizations
    viz_dir = 'selp_ellipsoid_visualizations'
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create subdirectories for each overlap and error range
    for overlap in ['large', 'small', 'none']:
        for error_range in ['small', 'medium', 'large']:
            os.makedirs(os.path.join(viz_dir, f"{overlap}_{error_range}"), exist_ok=True)
    
    # Save visualizations for selected pairs
    print("\nSaving ellipsoid visualizations...")
    for overlap in ['large', 'small', 'none']:
        for error_range in ['small', 'medium', 'large']:
            for pair_info in viz_pairs[overlap][error_range]:
                idx = pair_info['idx']
                error = pair_info['error']
                pair_data = test_data[idx]
                
                # Get predicted and ground truth rotations from saved data
                pred_rel_R = torch.tensor(pair_info['pred_rel_R'], device=device)
                gt_rel_R = torch.tensor(pair_info['gt_rel_R'], device=device)
                
                # Convert to Euler angles using the correct function
                gt_angles = rotation_matrix_to_euler_angles(gt_rel_R.cpu().numpy())
                pred_angles = rotation_matrix_to_euler_angles(pred_rel_R.cpu().numpy())
                
                # Save visualization
                save_ellipsoid(
                    test_data,
                    gt_angles,
                    pred_angles,
                    idx,
                    os.path.join(viz_dir, f"{overlap}_{error_range}")
                )
    
    print(f"\nVisualizations have been saved to {viz_dir}/")

    # integrated pngs
    def create_integrated_png_for_category(
        viz_dir, category, error_ranges=['small', 'medium', 'large'],
        images_per_range=10, output_dir='integrated_pngs'
    ):
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
            small_font = ImageFont.truetype("DejaVuSans.ttf", 24)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        # Collect images for each error range
        images_by_range = []
        errors_by_range = []
        for error_range in error_ranges:
            subdir = os.path.join(viz_dir, f"{category}_{error_range}")
            if not os.path.exists(subdir):
                images_by_range.append([])
                errors_by_range.append([])
                continue
            files = sorted([f for f in os.listdir(subdir) if f.endswith('.png')])
            images = [Image.open(os.path.join(subdir, f)) for f in files[:images_per_range]]
            images_by_range.append(images)
            
            # Get errors from the saved viz_pairs
            viz_pairs = np.load('selp_viz_pairs.npy', allow_pickle=True).item()
            errors = [pair['error'] for pair in viz_pairs[category][error_range][:images_per_range]]
            errors_by_range.append(errors)

        # Find max width and height
        img_width, img_height = (0, 0)
        for imgs in images_by_range:
            if imgs:
                img_width, img_height = imgs[0].size
                break

        n_cols = images_per_range
        n_rows = len(error_ranges)
        label_width = 280
        error_height = 40

        # Create a blank canvas
        integrated_img = Image.new(
            'RGB',
            (label_width + img_width * n_cols, (img_height + error_height) * n_rows),
            (255, 255, 255)
        )
        draw = ImageDraw.Draw(integrated_img)

        # Labels for each row
        row_labels = ["Small Error", "Medium Error", "Large Error"]

        for row, (label, imgs, errors) in enumerate(zip(row_labels, images_by_range, errors_by_range)):
            # Get text size for vertical centering
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            y = row * (img_height + error_height) + (img_height - text_height) // 2
            draw.text((20, y), label, fill="black", font=font)
            
            # Paste images and add error text
            for col, (img, error) in enumerate(zip(imgs, errors)):
                # Paste the image
                integrated_img.paste(
                    img, (label_width + col * img_width, row * (img_height + error_height))
                )
                # Add error text below the image
                error_text = f"{error:.2f}°"
                bbox = draw.textbbox((0, 0), error_text, font=small_font)
                text_width = bbox[2] - bbox[0]
                text_x = label_width + col * img_width + (img_width - text_width) // 2
                text_y = row * (img_height + error_height) + img_height + 5
                draw.text((text_x, text_y), error_text, fill="black", font=small_font)

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{category}_integrated.png")
        integrated_img.save(out_path)
        print(f"Saved integrated PNG for {category} at {out_path}")

    # Call for each category
    for category in ['large', 'small', 'none']:
        create_integrated_png_for_category(viz_dir, category)

    # Delete the individual visualization folder after integration
    shutil.rmtree(viz_dir)
    print(f"Deleted folder {viz_dir} and all its contents.")

if __name__ == '__main__':
    main() 
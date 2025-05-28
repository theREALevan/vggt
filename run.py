import os
import torch
import numpy as np
import cv2
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from visual_util import predictions_to_glb

# Configuration
IMAGE_FOLDER = '/home/yz864/vggt/images'   # folder of input images
OUT_GLTF     = os.path.join(IMAGE_FOLDER, 'scene.glb')  # output filename in images folder
CONF_THRESH  = 50.0                       # point‐cloud confidence threshold

# Setup device & model
device = "cuda" if torch.cuda.is_available() else "cpu"

dtype = torch.bfloat16 if (
    device=="cuda" and torch.cuda.get_device_capability()[0] >= 8
) else torch.float16

print(f"Device: {device}, AMP dtype: {dtype}")
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model.eval()

# Load & preprocess images or video
image_paths = []
for fn in os.listdir(IMAGE_FOLDER):
    if fn.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff','.webp')):
        image_paths.append(os.path.join(IMAGE_FOLDER, fn))
    elif fn.lower().endswith('.mp4'):
        # Handle video file
        video_path = os.path.join(IMAGE_FOLDER, fn)
        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 1)  # 1 frame per second
        
        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                frame_path = os.path.join(IMAGE_FOLDER, f"frame_{video_frame_num:06d}.png")
                cv2.imwrite(frame_path, frame)
                image_paths.append(frame_path)
                video_frame_num += 1
        vs.release()

if not image_paths:
    raise RuntimeError(f"No images or video found in {IMAGE_FOLDER}")

# Sort paths to ensure consistent ordering
image_paths = sorted(image_paths)

images = load_and_preprocess_images(image_paths).to(device)
H, W = images.shape[-2:]
print(f"Loaded {len(image_paths)} images of size {H}×{W}")


# Run inference
with torch.no_grad():
    print("Running VGGT inference...")
    with torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images)

# Decode camera poses (keep batch dim)
print("Decoding camera poses…")
extrinsic_t, intrinsic_t = pose_encoding_to_extri_intri(
    predictions["pose_enc"],     # shape (1, S, 9)
    image_size_hw=(H, W)
)
# Convert to NumPy and drop the batch axis
predictions["extrinsic"] = extrinsic_t.cpu().numpy().squeeze(0)  # (S, 3, 4)
predictions["intrinsic"] = intrinsic_t.cpu().numpy().squeeze(0)  # (S, 3, 3)

# Convert all other tensors to NumPy and drop batch dim
for k, v in list(predictions.items()):
    if isinstance(v, torch.Tensor):
        arr = v.cpu().numpy()        # (1, S, H, W, 1) or (1, S, H, W)
        predictions[k] = np.squeeze(arr, axis=0)
        print(f"Shape of {k}: {predictions[k].shape}")

# Unproject depth to world‐space points
print("Unprojecting depth maps to world points…")
predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
    predictions["depth"],         # (S, H, W, 1)
    predictions["extrinsic"],     # (S, 3, 4)
    predictions["intrinsic"],     # (S, 3, 3)
)
print(f"Shape of world_points_from_depth: {predictions['world_points_from_depth'].shape}")

# Ensure world_points_conf has the same shape as world_points_from_depth
if "world_points_conf" not in predictions:
    predictions["world_points_conf"] = np.ones_like(predictions["world_points_from_depth"][..., 0])
elif predictions["world_points_conf"].shape != predictions["world_points_from_depth"].shape[:-1]:
    predictions["world_points_conf"] = np.ones_like(predictions["world_points_from_depth"][..., 0])

# Load raw RGB images for coloring
print("Loading RGBs for point colors…")
# Use the same preprocessing as the demo
images = load_and_preprocess_images(image_paths).to(device)
# Convert to numpy
images = images.cpu().numpy()  # (S, 3, H, W)
# Convert from NCHW to NHWC format
images = np.transpose(images, (0, 2, 3, 1))  # (S, H, W, 3)
predictions["images"] = images
print(f"Shape of images after preprocessing: {predictions['images'].shape}")

# Build GLB scene (point cloud + cameras)
glbscene = predictions_to_glb(
    predictions,
    conf_thres=CONF_THRESH,
    filter_by_frames="All",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    target_dir=None,
    prediction_mode="Predicted Pointmap"
)

# Export to disk
print(f"Exporting to {OUT_GLTF} …")
glbscene.export(OUT_GLTF)
print(f"✔ Written {OUT_GLTF}")

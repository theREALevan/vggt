import numpy as np
import os
import json
from tqdm import tqdm

def load_categories():
    with open('categories.json', 'r') as f:
        return json.load(f)

def get_img_path(img_path, base_paths, categories):
    parts = img_path.split('/')
    if parts[0] == 'images':
        if len(parts) >= 4 and 'commons' in parts:
            commons_idx = parts.index('commons')
            if commons_idx + 1 < len(parts):
                scene_id1 = parts[1]
                scene_id2 = parts[2]
                scene_name = parts[commons_idx + 1]
                
                # Use categories mapping
                scene_id = f"{scene_id1}{scene_id2}"
                if scene_id in categories:
                    mapped_name = categories[scene_id]
                    scene_name = mapped_name
                
                # Get both underscore and space versions of the filename
                filename = parts[-1]
                filename_with_spaces = filename.replace('_', ' ')
                
                # Reconstruct the path using the mapped scene name
                new_parts = [parts[0], scene_name, 'commons', scene_name, '0', 'pictures']
                
                for base in base_paths:
                    # Try with original filename (with underscores)
                    candidate = os.path.join(base, *new_parts, filename)
                    if os.path.exists(candidate):
                        return candidate
                    
                    # Try with spaces in filename
                    candidate = os.path.join(base, *new_parts, filename_with_spaces)
                    if os.path.exists(candidate):
                        return candidate
    return None

def filter_valid_pairs(input_npy, output_npy, base_paths, pbar=None):
    print(f"\nProcessing {input_npy}...")
    data = np.load(input_npy, allow_pickle=True).item()
    categories = load_categories()
    
    valid_pairs = {}
    total_pairs = len(data)
    valid_count = 0
    invalid_path_format = 0
    missing_images = 0
    
    # List to store invalid addresses
    invalid_addresses = []
    
    print(f"Total pairs to check: {total_pairs}")
    
    # Create progress bar for this file
    file_pbar = tqdm(data.items(), 
                     desc=f"Filtering {os.path.basename(input_npy)}",
                     total=total_pairs,
                     leave=True)
    
    for key, pair_data in file_pbar:
        img1_path = get_img_path(pair_data['img1']['path'], base_paths, categories)
        img2_path = get_img_path(pair_data['img2']['path'], base_paths, categories)
        
        if img1_path is not None and img2_path is not None:
            # Create a copy of the pair data with full paths
            valid_pair = pair_data.copy()
            valid_pair['img1']['path'] = img1_path
            valid_pair['img2']['path'] = img2_path
            valid_pairs[key] = valid_pair
            valid_count += 1
        else:
            if not pair_data['img1']['path'].startswith('images/') or not pair_data['img2']['path'].startswith('images/'):
                invalid_path_format += 1
                if len(invalid_addresses) < 5:
                    # Replace underscores with spaces in filenames
                    img1_name = os.path.basename(pair_data['img1']['path']).replace('_', ' ')
                    img2_name = os.path.basename(pair_data['img2']['path']).replace('_', ' ')
                    invalid_addresses.append(f"Invalid format: {img1_name} - {img2_name}")
            else:
                missing_images += 1
                if len(invalid_addresses) < 5:
                    # Get original paths
                    img1_orig = pair_data['img1']['path']
                    img2_orig = pair_data['img2']['path']
                    
                    # Get reconstructed paths (what we tried to find)
                    parts1 = img1_orig.split('/')
                    parts2 = img2_orig.split('/')
                    if len(parts1) >= 4 and 'commons' in parts1 and len(parts2) >= 4 and 'commons' in parts2:
                        commons_idx1 = parts1.index('commons')
                        commons_idx2 = parts2.index('commons')
                        if commons_idx1 + 1 < len(parts1) and commons_idx2 + 1 < len(parts2):
                            scene_id1 = f"{parts1[1]}{parts1[2]}"
                            scene_id2 = f"{parts2[1]}{parts2[2]}"
                            scene_name1 = categories.get(scene_id1, parts1[commons_idx1 + 1])
                            scene_name2 = categories.get(scene_id2, parts2[commons_idx2 + 1])
                            
                            # Only replace underscores with spaces in filenames
                            filename1 = parts1[-1].replace('_', ' ')
                            filename2 = parts2[-1].replace('_', ' ')
                            
                            reconstructed1 = f"images/{scene_name1}/commons/{scene_name1}/0/pictures/{filename1}"
                            reconstructed2 = f"images/{scene_name2}/commons/{scene_name2}/0/pictures/{filename2}"
                            
                            invalid_addresses.append(
                                f"Missing images:\n"
                                f"  Image 1:\n"
                                f"    Original: {img1_orig}\n"
                                f"    Reconstructed: {reconstructed1}\n"
                                f"    Filename: {filename1}\n"
                                f"  Image 2:\n"
                                f"    Original: {img2_orig}\n"
                                f"    Reconstructed: {reconstructed2}\n"
                                f"    Filename: {filename2}"
                            )
        
        # Update progress bar with current stats
        file_pbar.set_postfix({
            'valid': valid_count,
            'invalid_format': invalid_path_format,
            'missing': missing_images
        })
        
        # Update overall progress if provided
        if pbar is not None:
            pbar.update(1)
    
    # Save invalid addresses to file
    if invalid_addresses:
        with open('invalid_addresses.txt', 'w') as f:
            for addr in invalid_addresses:
                f.write(addr + '\n\n')  # Add extra newline between entries
        print(f"\nSaved {len(invalid_addresses)} invalid addresses to invalid_addresses.txt")
    
    print(f"\nFiltering results for {input_npy}:")
    print(f"Total pairs: {total_pairs}")
    print(f"Valid pairs: {valid_count}")
    print(f"Filtered out: {total_pairs - valid_count}")
    print(f"  - Invalid path format: {invalid_path_format}")
    print(f"  - Missing image files: {missing_images}")
    
    # Save filtered data with full paths
    np.save(output_npy, valid_pairs)
    print(f"Saved valid pairs to {output_npy}")
    
    return total_pairs, valid_count

def main():
    # Base paths to search
    base_paths = [
        '/share/phoenix/nfs05/S8/jt664/WikiSFM/data/main',
        '/share/phoenix/nfs06/S9/jt664/megascenes_local/data'
    ]
    
    # Load both files to get total count
    overlap_data = np.load('metadata/train_overlap_megascenes_path.npy', allow_pickle=True).item()
    none_data = np.load('metadata/train_none_megascenes_path.npy', allow_pickle=True).item()
    total_pairs = len(overlap_data) + len(none_data)
    
    print(f"\nStarting filtering process...")
    print(f"Total pairs to process: {total_pairs}")
    
    # Create overall progress bar
    with tqdm(total=total_pairs, desc="Overall Progress", position=0) as pbar:
        # Process overlapping pairs
        overlap_total, overlap_valid = filter_valid_pairs(
            'metadata/train_overlap_megascenes_path.npy',
            'metadata/train_overlap_megascenes_path_valid.npy',
            base_paths,
            pbar
        )
        
        # Process non-overlapping pairs
        none_total, none_valid = filter_valid_pairs(
            'metadata/train_none_megascenes_path.npy',
            'metadata/train_none_megascenes_path_valid.npy',
            base_paths,
            pbar
        )
    
    # Print final summary
    print("\nFinal Summary:")
    print(f"Overlapping pairs: {overlap_valid}/{overlap_total} valid")
    print(f"Non-overlapping pairs: {none_valid}/{none_total} valid")
    print(f"Total valid pairs: {overlap_valid + none_valid}/{total_pairs}")

if __name__ == '__main__':
    main() 
"""
Utility script to add zero padding to ground truth data for model training.

This script pads ground truth data arrays from 44x44 to 64x64 to match
the model's input size requirements. It processes all numpy files in a
specified directory and applies consistent padding.
"""

import os

import cv2
import numpy as np
from natsort import natsorted


def pad_array(data, target_size=64):
    """
    Pad a numpy array with zeros to reach the target size.
    
    Args:
        data (np.ndarray): Input array to pad
        target_size (int): Target size for width and height dimensions
        
    Returns:
        np.ndarray: Padded array
    """
    original_size = data.shape[1]  # Assuming square input
    pad_size = (target_size - original_size) // 2
    
    padded_array = np.pad(
        data, 
        pad_width=((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 
        mode='constant', 
        constant_values=0
    )
    
    return padded_array


def process_directory(directory_path, save=False, visualize=False):
    """
    Process all numpy files in a directory by adding zero padding.
    
    Args:
        directory_path (str): Path to directory containing numpy files
        save (bool): Whether to save the padded arrays back to files
        visualize (bool): Whether to save visualizations of the padded arrays
        
    Returns:
        int: Number of files processed
    """
    gt_files = natsorted(os.listdir(directory_path))
    count = 0
    
    for gt_file in gt_files:
        if not gt_file.endswith('.npy'):
            continue
        file_path = os.path.join(directory_path, gt_file)
        data = np.load(file_path)
        
        padded_array = pad_array(data)
        
        if visualize:
            visualization_dir = os.path.join(directory_path, "visualizations")
            os.makedirs(visualization_dir, exist_ok=True)
            
            for i, img in enumerate(np.asarray(padded_array)):
                # Normalize the outputs to 255 in each pixel
                output = img * 255
                # Duplicate it to be an RGB image
                output = np.repeat(output[:, :, np.newaxis], 3, axis=2)
                # Save it as a cv2 image
                cv2.imwrite(os.path.join(visualization_dir, f"{gt_file}_{i}.png"), output)
        
        if save:
            np.save(file_path, padded_array)
            
        count += 1
        
    return count


if __name__ == "__main__":
    # Configuration
    DIR_PATH = "/home/cc/updated_gt_maps"
    SAVE_PADDED = False  # Set to True to overwrite original files
    VISUALIZE = False    # Set to True to save visualizations
    
    # Process files
    num_processed = process_directory(DIR_PATH, save=SAVE_PADDED, visualize=VISUALIZE)
    print(f"Processed {num_processed} files.")
    print(f"Files {'saved' if SAVE_PADDED else 'processed but not saved'}.")
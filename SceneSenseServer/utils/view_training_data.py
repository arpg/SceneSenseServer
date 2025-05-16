"""
Utility script for visualizing training data.

This script loads and visualizes training data for the diffusion model,
allowing verification of the dataset before training.
"""

import os

import numpy as np
import open3d as o3d

from SceneSenseServer.utils import utils


def load_training_data(data_dir, sample_idx):
    """
    Load training data from numpy files.
    
    Args:
        data_dir (str): Directory containing the training data
        sample_idx (int): Index of the sample to load
        
    Returns:
        tuple: (input_data, latent_data, inpainting_target, inpainting_unocc, bounds)
    """
    # Base directory for data
    base_dir = f"{data_dir}/{sample_idx}"
    
    # Load data files
    latents = np.load(f"{base_dir}_latents.npy")
    inpainting_target = np.load(f"{base_dir}_inpainting_target.npy")
    inpainting_unocc = np.load(f"{base_dir}_inpainting_unocc.npy")
    input_data = np.load(f"{base_dir}_input_data.npy")
    bounds = np.load(f"{base_dir}_bounds.npy")
    
    return input_data, latents, inpainting_target, inpainting_unocc, bounds


def visualize_point_clouds(input_data, inpainting_target, bounds):
    """
    Visualize the input data and inpainting target as point clouds.
    
    Args:
        input_data (np.ndarray): Input point map data
        inpainting_target (np.ndarray): Target point map data
        bounds (np.ndarray): Bounds for visualization
    """
    # Convert input data to point cloud
    input_pcd = utils.pointmap_to_pc(input_data, 0.03125, bounds=bounds)
    input_pcd = utils.set_rgb(input_pcd)
    
    # Convert inpainting target to point cloud
    target_pcd = utils.pointmap_to_pc(inpainting_target, 0.03125, bounds=bounds)
    target_pcd = utils.set_rgb(target_pcd, 0)  # Red color for target
    
    # Create coordinate frame for reference
    coor = o3d.geometry.TriangleMesh.create_coordinate_frame()
    
    # Visualize
    o3d.visualization.draw_geometries([input_pcd, target_pcd, coor])


def main():
    """Main function to run the training data visualization."""
    # Configuration
    DATA_DIR = "data/diff_data/diff_data"
    SAMPLE_IDX = 200
    
    # Load training data
    input_data, latents, inpainting_target, inpainting_unocc, bounds = load_training_data(
        DATA_DIR, 
        SAMPLE_IDX
    )
    
    # Print data shapes for debugging
    print(f"Input data shape: {input_data.shape}")
    print(f"Latents shape: {latents.shape}")
    print(f"Inpainting target shape: {inpainting_target.shape}")
    print(f"Inpainting unocc shape: {inpainting_unocc.shape}")
    print(f"Bounds: {bounds}")
    
    # Visualize data as point clouds
    visualize_point_clouds(input_data, inpainting_target, bounds)


if __name__ == "__main__":
    main()
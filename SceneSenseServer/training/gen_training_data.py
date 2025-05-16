"""
Training data generation script for diffusion models.

This script processes point cloud data to create training datasets for diffusion models.
It transforms point clouds using odometry data and extracts local pointmaps at each pose.
"""

import copy
import os

import numpy as np
import open3d as o3d
from natsort import natsorted
from scipy.spatial.transform import Rotation

from SceneSenseServer.utils import utils


def generate_training_data(data_path, full_pcd_path, odom_path, output_path, radius=3.2, voxel_size=0.1):
    """
    Generate training data by transforming a full point cloud to different odometry poses.
    
    Args:
        data_path (str): Base data directory.
        full_pcd_path (str): Path to the full point cloud file.
        odom_path (str): Path to the directory containing odometry files.
        output_path (str): Path to save the generated training data.
        radius (float): Radius for local point extraction (default: 3.2).
        voxel_size (float): Size of voxels for pointmap generation (default: 0.1).
        
    Returns:
        int: Number of generated training samples.
    """
    # Load the full point cloud
    full_pcd = o3d.io.read_point_cloud(full_pcd_path)
    
    # Get sorted list of odometry files
    odom_file_names = natsorted(os.listdir(odom_path))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    count = 0
    for idx, odom_file in enumerate(odom_file_names):
        # Load odometry data
        pose = np.load(os.path.join(odom_path, odom_file))
        
        # Create transformation matrix
        rotation_obj = Rotation.from_rotvec(pose[3:])
        hm_tx_mat = utils.homogeneous_transform(pose[0:3], rotation_obj.as_quat())
        
        # Transform point cloud
        tx_pcd = copy.deepcopy(full_pcd)
        tx_pcd.transform(utils.inverse_homogeneous_transform(hm_tx_mat))
        
        # Extract local points
        local_points = utils.points_within_distance(0.0, 0.0, np.asarray(tx_pcd.points), radius)
        local_pcd = o3d.geometry.PointCloud()
        local_pcd.points = o3d.utility.Vector3dVector(local_points)
        local_pcd = utils.update_points(local_pcd, -1.3, 1, 2)
        
        # Convert to pointmap
        local_octomap_pm = utils.pc_to_pointmap(
            np.asarray(local_pcd.points),
            voxel_size=voxel_size,
            x_y_bounds=[-radius, radius],
            z_bounds=[-1.3, 1.0]
        )
        
        # Save the pointmap
        save_path = os.path.join(output_path, f"{idx}.npy")
        np.save(save_path, local_octomap_pm)
        
        print(f"Generated sample {idx}, shape: {local_octomap_pm.shape}")
        count += 1
    
    return count


if __name__ == "__main__":
    # Configuration - replace these with your actual paths
    DATA_PATH = "/path/to/data"
    FULL_PCD_PATH = os.path.join(DATA_PATH, "full_map.pcd")
    ODOM_PATH = os.path.join(DATA_PATH, "odometry/odometry/")
    OUTPUT_PATH = os.path.join(DATA_PATH, "training_data/trajectory_01/")
    
    # Generate the training data
    num_samples = generate_training_data(
        data_path=DATA_PATH,
        full_pcd_path=FULL_PCD_PATH,
        odom_path=ODOM_PATH,
        output_path=OUTPUT_PATH
    )
    
    print(f"Successfully generated {num_samples} training samples in {OUTPUT_PATH}")
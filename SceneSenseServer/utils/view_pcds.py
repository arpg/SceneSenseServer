"""
Utility script for visualizing point clouds with odometry transformation.

This script loads point clouds, applies odometry-based transformations,
and visualizes the result with a coordinate frame for reference.
"""

import os

import numpy as np
import open3d as o3d
from natsort import natsorted
from scipy.spatial.transform import Rotation

from SceneSenseServer.utils import utils


def load_point_cloud(file_path):
    """
    Load a point cloud file and perform initial preprocessing.
    
    Args:
        file_path (str): Path to the point cloud file
        
    Returns:
        o3d.geometry.PointCloud: Processed point cloud
    """
    pcd = o3d.io.read_point_cloud(file_path)
    # Apply height filtering
    pcd = utils.update_points(pcd, -1.3, 1, 2)
    return pcd


def apply_odometry_transform(pcd, odom_path, odom_idx=480):
    """
    Apply odometry-based transformation to the point cloud.
    
    Args:
        pcd (o3d.geometry.PointCloud): Point cloud to transform
        odom_path (str): Path to odometry data directory
        odom_idx (int): Index of the odometry file to use
        
    Returns:
        tuple: (transformed_point_cloud, coordinate_frame)
    """
    # Create coordinate frame for visualization
    coor = o3d.geometry.TriangleMesh.create_coordinate_frame()
    
    # Load odometry data
    odom_file_names = natsorted(os.listdir(odom_path))
    pose = np.load(os.path.join(odom_path, odom_file_names[odom_idx]))
    
    # Create homogeneous transformation matrix
    rotation_obj = Rotation.from_rotvec(pose[3:])
    hm_tx_mat = utils.homogeneous_transform(pose[0:3], rotation_obj.as_quat())
    
    # Apply transformation to the point cloud
    pcd.transform(utils.inverse_homogeneous_transform(hm_tx_mat))
    
    # Additional filtering to remove distant points
    pcd = utils.update_points(pcd, -100, 2, 1)
    
    return pcd, coor


def main():
    """Main function to run the point cloud visualization."""
    # Configuration
    POINTS_PATH = "data/range_max/running_occ.pcd"
    ODOM_PATH = "/hdd/spot_diff_data/odometry/odometry"
    ODOM_IDX = 480
    
    # Load point cloud
    pcd = load_point_cloud(POINTS_PATH)
    
    # Apply odometry transformation
    pcd, coor = apply_odometry_transform(pcd, ODOM_PATH, ODOM_IDX)
    
    # Visualize
    o3d.visualization.draw_geometries([pcd, coor])


if __name__ == "__main__":
    main()
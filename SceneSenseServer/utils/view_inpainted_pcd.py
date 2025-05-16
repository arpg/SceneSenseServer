"""
Utility script for visualizing original and inpainted point clouds.

This script loads original and inpainted point clouds, transforms them based on
odometry data, and visualizes them together for comparison.
"""

import os

import numpy as np
import open3d as o3d
from natsort import natsorted
from scipy.spatial.transform import Rotation

from SceneSenseServer.utils import utils


def load_and_preprocess_point_clouds(points_path, unoc_points_path):
    """
    Load and preprocess the original point clouds.
    
    Args:
        points_path (str): Path to the occupied points PCD file
        unoc_points_path (str): Path to the unoccupied points PCD file
        
    Returns:
        tuple: (occupied_point_cloud, unoccupied_point_cloud)
    """
    # Load point clouds
    pcd = o3d.io.read_point_cloud(points_path)
    unoc_pcd = o3d.io.read_point_cloud(unoc_points_path)
    
    # Apply initial height filtering
    pcd = utils.update_points(pcd, -1.3, 1, 2)
    unoc_pcd = utils.update_points(unoc_pcd, -1.3, 1, 2)
    
    return pcd, unoc_pcd


def apply_odometry_transform(pcd, unoc_pcd, odom_path, odom_idx=480, frontier_offset=None):
    """
    Apply odometry-based transformation to point clouds.
    
    Args:
        pcd (o3d.geometry.PointCloud): Occupied points point cloud
        unoc_pcd (o3d.geometry.PointCloud): Unoccupied points point cloud
        odom_path (str): Path to the odometry data directory
        odom_idx (int): Index of the odometry file to use
        frontier_offset (np.ndarray, optional): Additional translation to apply
        
    Returns:
        tuple: (transformed_pcd, transformed_unoc_pcd, coordinate_frame)
    """
    # Create coordinate frame for visualization
    coor = o3d.geometry.TriangleMesh.create_coordinate_frame()
    
    # Load odometry data
    odom_file_names = natsorted(os.listdir(odom_path))
    pose = np.load(os.path.join(odom_path, odom_file_names[odom_idx]))
    
    # Create transformation matrix
    rotation_obj = Rotation.from_rotvec(pose[3:])
    hm_tx_mat = utils.homogeneous_transform(pose[0:3], rotation_obj.as_quat())
    
    # Apply odometry transformation
    pcd.transform(utils.inverse_homogeneous_transform(hm_tx_mat))
    unoc_pcd.transform(utils.inverse_homogeneous_transform(hm_tx_mat))
    
    # Clean up points
    pcd = utils.update_points(pcd, -100, 2, 1)
    unoc_pcd = utils.update_points(unoc_pcd, -100, 2, 1)
    
    # Apply additional frontier transformation if provided
    if frontier_offset is not None:
        frontier_trans = utils.homogeneous_transform(
            frontier_offset, 
            np.array([0, 0, 0, 1])
        )
        pcd.transform(utils.inverse_homogeneous_transform(frontier_trans))
        unoc_pcd.transform(utils.inverse_homogeneous_transform(frontier_trans))
    
    return pcd, unoc_pcd, coor


def extract_local_points(pcd, unoc_pcd, radius=2.0, max_height=1.0):
    """
    Extract local points around the origin within specified constraints.
    
    Args:
        pcd (o3d.geometry.PointCloud): Occupied points point cloud
        unoc_pcd (o3d.geometry.PointCloud): Unoccupied points point cloud
        radius (float): Radius around the origin to extract points
        max_height (float): Maximum height for points
        
    Returns:
        tuple: (local_pcd, local_unoc_pcd)
    """
    # Extract local points
    local_points = utils.points_within_distance(0.0, 0.0, np.asarray(pcd.points), radius)
    local_points = local_points[local_points[:, 2] < max_height]
    
    local_points_unoc = utils.points_within_distance(0.0, 0.0, np.asarray(unoc_pcd.points), radius)
    local_points_unoc = local_points_unoc[local_points_unoc[:, 2] < max_height]
    
    # Create point clouds
    local_pcd = o3d.geometry.PointCloud()
    local_pcd.points = o3d.utility.Vector3dVector(local_points)
    local_pcd = utils.set_rgb(local_pcd)
    
    local_unoc_pcd = o3d.geometry.PointCloud()
    local_unoc_pcd.points = o3d.utility.Vector3dVector(local_points_unoc)
    
    return local_pcd, local_unoc_pcd


def visualize_point_clouds(local_pcd, inpainted_pcd_path, coor):
    """
    Visualize the original and inpainted point clouds together.
    
    Args:
        local_pcd (o3d.geometry.PointCloud): Local point cloud
        inpainted_pcd_path (str): Path to the inpainted point cloud file
        coor (o3d.geometry.TriangleMesh): Coordinate frame for visualization
    """
    # Load inpainted point cloud
    diff_points = o3d.io.read_point_cloud(inpainted_pcd_path)
    diff_points = utils.set_rgb(diff_points, 0)  # Set color to red
    
    # Visualize
    o3d.visualization.draw_geometries([local_pcd, diff_points, coor])


def main():
    """Main function to run the point cloud visualization."""
    # Configuration
    POINTS_PATH = "data/range_max/running_occ.pcd"
    UNOC_POINTS_PATH = "data/range_max/test_unoc.pcd"
    ODOM_PATH = "/hdd/spot_diff_data/odometry/odometry"
    INPAINTED_PCD_PATH = "test_inpainted_pcd.pcd"
    FRONTIER_OFFSET = np.array([1.0, 0.0, 0])
    
    # Load and preprocess point clouds
    pcd, unoc_pcd = load_and_preprocess_point_clouds(POINTS_PATH, UNOC_POINTS_PATH)
    
    # Apply transformations
    pcd, unoc_pcd, coor = apply_odometry_transform(
        pcd, 
        unoc_pcd, 
        ODOM_PATH, 
        frontier_offset=FRONTIER_OFFSET
    )
    
    # Extract local points
    local_pcd, _ = extract_local_points(pcd, unoc_pcd)
    
    # Visualize
    visualize_point_clouds(local_pcd, INPAINTED_PCD_PATH, coor)


if __name__ == "__main__":
    main()
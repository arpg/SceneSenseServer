"""
Utility script for visualizing OctoMap output point clouds.

This script loads and visualizes point clouds generated from OctoMap files,
coloring occupied and free space differently for better visualization.
"""

import os

import numpy as np
import open3d as o3d


def load_point_cloud(file_path):
    """
    Load a point cloud from a file.
    
    Args:
        file_path (str): Path to the point cloud file
        
    Returns:
        o3d.geometry.PointCloud: Loaded point cloud
    """
    # Load point cloud file
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd


def visualize_point_clouds(pcd, free_pcd=None):
    """
    Visualize one or more point clouds.
    
    Args:
        pcd (o3d.geometry.PointCloud): Main point cloud (occupied space)
        free_pcd (o3d.geometry.PointCloud, optional): Free space point cloud
    """
    # Create coordinate frame for reference
    coor = o3d.geometry.TriangleMesh.create_coordinate_frame()
    
    # Prepare geometries to visualize
    geometries = [pcd, coor]
    if free_pcd is not None:
        geometries.insert(1, free_pcd)
    
    # Display point clouds
    o3d.visualization.draw_geometries(geometries)


def main():
    """Main function to run the OctoMap visualization."""
    # Configuration
    OCTOMAP_DIR = "octomap"
    PCD_FILE = "cloud.pcd"
    FREE_PCD_FILE = "cloud_free.pcd"
    
    # File paths
    pcd_path = os.path.join(OCTOMAP_DIR, PCD_FILE)
    free_pcd_path = os.path.join(OCTOMAP_DIR, FREE_PCD_FILE)
    
    # Load point clouds
    pcd = load_point_cloud(pcd_path)
    
    # Set color for occupied points (blue)
    pcd.paint_uniform_color([0, 0, 1])
    
    # Check if free space file exists
    if os.path.exists(free_pcd_path):
        free_pcd = load_point_cloud(free_pcd_path)
        # Set color for free space points (green)
        free_pcd.paint_uniform_color([0, 1, 0])
        visualize_point_clouds(pcd, free_pcd)
    else:
        visualize_point_clouds(pcd)


if __name__ == "__main__":
    main()
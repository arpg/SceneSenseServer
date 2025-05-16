"""
Corner image generation script for visualization.

This script generates and visualizes corner images from point cloud data,
comparing the original point cloud with diffused point cloud data for visualization.
"""

import copy
import os

import numpy as np
import open3d as o3d
from natsort import natsorted
from scipy.spatial.transform import Rotation

from SceneSenseServer.utils import utils


def remove_close_points(pcd, diff_pcd, threshold=0.1):
    """
    Remove points in diff_pcd that are close to points in pcd.
    
    Args:
        pcd (o3d.geometry.PointCloud): Reference point cloud.
        diff_pcd (o3d.geometry.PointCloud): Point cloud to filter.
        threshold (float): Distance threshold for point removal.
        
    Returns:
        np.ndarray: Filtered points from diff_pcd.
    """
    # Convert Open3D point clouds to numpy arrays
    pcd_points = np.asarray(pcd.points)
    diff_pcd_points = np.asarray(diff_pcd.points)

    # Create a KDTree for the pcd points
    pcd_kdtree = o3d.geometry.KDTreeFlann(pcd)

    # List to hold indices of points to remove
    indices_to_remove = []

    # Iterate through each point in diff_pcd
    for i, point in enumerate(diff_pcd_points):
        [k, idx, _] = pcd_kdtree.search_radius_vector_3d(point, threshold)
        if k > 0:
            indices_to_remove.append(i)

    # Remove points from diff_pcd
    diff_pcd_points_filtered = np.delete(diff_pcd_points, indices_to_remove, axis=0)
    return diff_pcd_points_filtered


def remove_points_within_distance(points, distance):
    """
    Remove points within a certain distance from the origin (0,0,0).
    
    Args:
        points (np.ndarray): An nx3 array of 3D points.
        distance (float): The distance threshold.
        
    Returns:
        np.ndarray: The filtered array of 3D points.
    """
    # Compute the squared distance from the origin for each point
    squared_distances = np.sum(points**2, axis=1)
    
    # Compute the squared distance threshold to avoid computing square root
    squared_distance_threshold = distance**2
    
    # Filter points based on the squared distance
    filtered_points = points[squared_distances > squared_distance_threshold]
    
    return filtered_points


def prepare_pcd_data(points_path, diff_points_path, transform=None, filter_distance=0.6):
    """
    Prepare and filter point cloud data for visualization.
    
    Args:
        points_path (str): Path to the original point cloud directory.
        diff_points_path (str): Path to the diffused point cloud directory.
        transform (np.ndarray, optional): Transformation matrix to apply to the point clouds.
        filter_distance (float): Distance from origin to filter points.
        
    Returns:
        tuple: (original_pcd_voxel, diffused_pcd_voxel, combined_pcd_voxel, coordinate_frame)
    """
    # Create coordinate frame for visualization
    coor = o3d.geometry.TriangleMesh.create_coordinate_frame()
    coor.scale(0.5, center=coor.get_center())
    
    # Define transformation if not provided
    if transform is None:
        transform = utils.homogeneous_transform(np.array([9.5, 1.0, 0.3]), np.array([0, 0, 0, 1]))
    
    # Load and process original point cloud
    point_file_names = natsorted(os.listdir(points_path))
    pcd = o3d.io.read_point_cloud(os.path.join(points_path, point_file_names[1]))
    pcd.transform(utils.inverse_homogeneous_transform(transform))
    
    # Filter points
    points = np.asarray(pcd.points)
    points = points[points[:, 2] < 1.0]
    points = points[points[:, 2] > -1.2]
    points = remove_points_within_distance(points, filter_distance)
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Color the original point cloud
    pcd_colors = np.zeros((len(np.asarray(pcd.points)), 3))
    max_z = np.max(np.asarray(pcd.points)[:, 2], axis=0)
    min_z = np.min(np.asarray(pcd.points)[:, 2], axis=0)
    pcd_colors[:, 1] = ((np.asarray(pcd.points)[:, 2] - min_z) / (max_z - min_z)) * (1 - 0.3) + 0.3
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    pcd_vox = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.15)
    
    # Load and process diffused point cloud
    diff_point_file_names = natsorted(os.listdir(diff_points_path))
    pcd_diff = o3d.io.read_point_cloud(os.path.join(diff_points_path, diff_point_file_names[-10]))
    pcd_diff.transform(utils.inverse_homogeneous_transform(transform))
    
    # Filter diffused points
    points = np.asarray(pcd_diff.points)
    points = points[points[:, 2] < 1.0]
    points = points[points[:, 2] > -1.2]
    points = utils.points_within_distance(0, 0, points, 3.3)
    points = remove_points_within_distance(points, filter_distance)
    pcd_diff.points = o3d.utility.Vector3dVector(points)
    
    # Remove points that are close to original points
    diff_point_filtered = remove_close_points(pcd, pcd_diff)
    diff_pcd_filtered = o3d.geometry.PointCloud()
    diff_pcd_filtered.points = o3d.utility.Vector3dVector(diff_point_filtered)
    
    # Color the diffused point cloud
    colors = np.zeros((len(np.asarray(diff_pcd_filtered.points)), 3))
    max_z = np.max(np.asarray(diff_pcd_filtered.points)[:, 2], axis=0)
    min_z = np.min(np.asarray(diff_pcd_filtered.points)[:, 2], axis=0)
    colors[:, 0] = ((np.asarray(diff_pcd_filtered.points)[:, 2] - min_z) / (max_z - min_z)) * (1 - 0.3) + 0.3
    diff_pcd_filtered.colors = o3d.utility.Vector3dVector(colors)
    
    pcd_diff_vox = o3d.geometry.VoxelGrid.create_from_point_cloud(diff_pcd_filtered, voxel_size=0.15)
    
    # Combine original and diffused points
    pcd_points = np.asarray(copy.deepcopy(pcd).points)
    diff_points = np.asarray(diff_pcd_filtered.points)
    combined_points = np.append(pcd_points, diff_points, axis=0)
    combined_colors = np.append(pcd_colors, colors, axis=0)
    
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(combined_points)
    final_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    combined_vox = o3d.geometry.VoxelGrid.create_from_point_cloud(final_pcd, voxel_size=0.15)
    
    return pcd_vox, pcd_diff_vox, combined_vox, coor


def create_interactive_visualizer(pcd_vox, pcd_diff_vox, coor):
    """
    Create an interactive visualizer to toggle between point cloud views.
    
    Args:
        pcd_vox (o3d.geometry.VoxelGrid): Original point cloud voxels.
        pcd_diff_vox (o3d.geometry.VoxelGrid): Diffused point cloud voxels.
        coor (o3d.geometry.TriangleMesh): Coordinate frame for visualization.
        
    Returns:
        o3d.visualization.VisualizerWithKeyCallback: Configured visualizer.
    """
    # Define callbacks for keyboard interaction
    def show_original(vis):
        """Show original point cloud on 'D' key press."""
        ctr = vis.get_view_control()
        view_param = ctr.convert_to_pinhole_camera_parameters()
        vis.clear_geometries()
        vis.add_geometry(pcd_vox)
        vis.add_geometry(coor)
        ctr.convert_from_pinhole_camera_parameters(view_param)
        return False
    
    def show_combined(vis):
        """Show combined point clouds on 'A' key press."""
        ctr = vis.get_view_control()
        view_param = ctr.convert_to_pinhole_camera_parameters()
        vis.clear_geometries()
        vis.add_geometry(pcd_diff_vox)
        vis.add_geometry(pcd_vox)
        vis.add_geometry(coor)
        ctr.convert_from_pinhole_camera_parameters(view_param)
        return False
    
    # Create and configure visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(pcd_vox)
    vis.add_geometry(coor)
    
    # Register key callbacks
    vis.register_key_callback(68, show_original)  # 'D' key
    vis.register_key_callback(65, show_combined)  # 'A' key
    
    return vis


def main():
    """Main function to generate and visualize corner images."""
    # Configuration - update these paths to your actual data directories
    points_path = "/hdd/IJRR_data/corner_maps/baseline_pcds"
    diff_points_path = "/hdd/IJRR_data/corner_maps/diffused_pcds"
    
    # Create transformation for visualization
    frontier_trans = utils.homogeneous_transform(np.array([9.5, 1.0, 0.3]), np.array([0, 0, 0, 1]))
    
    # Prepare the point cloud data
    pcd_vox, pcd_diff_vox, combined_vox, coor = prepare_pcd_data(
        points_path, 
        diff_points_path,
        transform=frontier_trans
    )
    
    # Create and run the interactive visualizer
    vis = create_interactive_visualizer(pcd_vox, pcd_diff_vox, coor)
    print("\nVisualization controls:")
    print("Press 'D' to view the original point cloud")
    print("Press 'A' to view the combined point cloud (original + diffused)")
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
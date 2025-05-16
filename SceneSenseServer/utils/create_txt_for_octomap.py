"""
Utility script for creating text files compatible with OctoMap from point cloud data.

This script converts point cloud data from Open3D format to text files that can be
processed by OctoMap tools like octomap_server.
"""

import os
import time

import numpy as np
import open3d as o3d

from SceneSenseServer.utils import utils


def load_point_cloud(input_path):
    """
    Load a point cloud file.
    
    Args:
        input_path (str): Path to the point cloud file
        
    Returns:
        o3d.geometry.PointCloud: Loaded point cloud
    """
    print(f"Loading point cloud from: {input_path}")
    pcd = o3d.io.read_point_cloud(input_path)
    return pcd


def write_point_cloud_to_text(pcd, output_path, include_color=False):
    """
    Write point cloud data to a text file compatible with OctoMap.
    
    Args:
        pcd (o3d.geometry.PointCloud): Point cloud to convert
        output_path (str): Path to save the output text file
        include_color (bool): Whether to include color information
        
    Returns:
        int: Number of points written
    """
    points = np.asarray(pcd.points)
    
    if include_color and pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        colors = None
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for i in range(len(points)):
            if colors is not None:
                # Format: x y z r g b
                color_values = [int(c * 255) for c in colors[i]]
                f.write(f"{points[i][0]} {points[i][1]} {points[i][2]} "
                        f"{color_values[0]} {color_values[1]} {color_values[2]}\n")
            else:
                # Format: x y z
                f.write(f"{points[i][0]} {points[i][1]} {points[i][2]}\n")
    
    print(f"Wrote {len(points)} points to {output_path}")
    return len(points)


def process_point_cloud(input_path, output_path, include_color=False, height_filter=None):
    """
    Process a point cloud and save it as a text file for OctoMap.
    
    Args:
        input_path (str): Path to input point cloud file
        output_path (str): Path to save output text file
        include_color (bool): Whether to include color information
        height_filter (tuple, optional): Tuple of (min_height, max_height) to filter points
        
    Returns:
        int: Number of points written
    """
    pcd = load_point_cloud(input_path)
    
    if height_filter:
        min_height, max_height = height_filter
        pcd = utils.update_points(pcd, min_height, max_height, 2)
        print(f"Applied height filter: {min_height} to {max_height}")
    
    return write_point_cloud_to_text(pcd, output_path, include_color)


def main():
    """Main function to convert point cloud files to OctoMap-compatible text files."""
    # Configuration
    INPUT_PATH = "input.pcd"
    OUTPUT_PATH = "octomap/cloud.txt"
    HEIGHT_FILTER = (-1.3, 1.0)  # Filter points between -1.3 and 1.0 meters in height
    INCLUDE_COLOR = False
    
    # Process time tracking
    start_time = time.time()
    
    # Process the point cloud
    num_points = process_point_cloud(
        INPUT_PATH, 
        OUTPUT_PATH, 
        include_color=INCLUDE_COLOR,
        height_filter=HEIGHT_FILTER
    )
    
    # Print processing summary
    elapsed_time = time.time() - start_time
    print(f"Processed {num_points} points in {elapsed_time:.2f} seconds")
    print(f"Output saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

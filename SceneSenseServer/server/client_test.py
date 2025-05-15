"""
Client for testing the diffusion server.

This script demonstrates how to interact with the diffusion server by:
1. Testing the basic connection to the server
2. Preparing and sending point cloud data for processing
3. Receiving and visualizing the processed results
"""

import requests
from flask import Flask

# URL of the Flask server (assuming the server is running on localhost and port 5000)
#this just gets the basic response from the server, could be good for testing connections
url = "http://localhost:5000/"

# Send a GET request to the server
response = requests.get(url)
#there is also the post call
# Check if the request was successful
if response.status_code == 200:
    print('Response from server:', response.text)
else:
    print('Failed to retrieve data:', response.status_code)


post_url = "http://localhost:5000/diffusion"


import copy
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass

import cv2
import numpy as np

#get local occ info
import open3d as o3d
import torch
import torch.nn.functional as F
import wandb
from cleanfid import fid
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from huggingface_hub import login
from natsort import natsorted
from scipy.spatial.transform import Rotation
from spconv.pytorch.utils import PointToVoxel
from tqdm.auto import tqdm

import SceneSenseServer.utils.utils as utils

# Server connection configuration
BASE_URL = "http://localhost:5000"
ROOT_ENDPOINT = f"{BASE_URL}/"
DIFFUSION_ENDPOINT = f"{BASE_URL}/diffusion"

def test_server_connection():
    """Test the basic connection to the server."""
    response = requests.get(ROOT_ENDPOINT)
    
    if response.status_code == 200:
        print('Server connection successful. Response:', response.text)
        return True
    else:
        print('Failed to connect to server:', response.status_code)
        return False

def prepare_point_cloud_data():
    """
    Load and prepare point cloud data for the diffusion server.
    
    Returns:
        tuple: Contains local_octomap_pm, local_octomap_pm_unon (occupied and unoccupied pointmaps)
    """
    # Load point cloud data
    # points_path = "data/range_max/running_occ.pcd"
    # unoc_points = "data/range_max/test_unoc.pcd"
    points_path = "/hdd/spot_diff_data/point_clouds/point_clouds/10.pcd"
    unoc_points = "/hdd/spot_diff_data/point_clouds/point_clouds/10.pcd"
    
    pcd = o3d.io.read_point_cloud(points_path)
    pcd = utils.update_points(pcd, -1.3, 1, 2)
    unoc_pcd = o3d.io.read_point_cloud(unoc_points)
    unoc_pcd = utils.update_points(unoc_pcd, -1.3, 1, 2)

    # Create coordinate frame for visualization
    coor = o3d.geometry.TriangleMesh.create_coordinate_frame()

    # Load and apply odometry transformation
    odom_path = "/media/brendan/brendan_ssd/odometry/odometry"
    odom_file_names = natsorted(os.listdir(odom_path))
    pose = np.load(os.path.join(odom_path, odom_file_names[480]))
    rotation_obj = Rotation.from_rotvec(pose[3::])
    hm_tx_mat = utils.homogeneous_transform(pose[0:3], rotation_obj.as_quat())
    
    # Apply frontier transformation
    frontier_trans = utils.homogeneous_transform(np.array([3.0, -2.0, 0]), np.array([0, 0, 0, 1]))
    
    # Transform point clouds
    pcd.transform(utils.inverse_homogeneous_transform(hm_tx_mat))
    unoc_pcd.transform(utils.inverse_homogeneous_transform(hm_tx_mat))
    
    # Clean up spurious detections
    pcd = utils.update_points(pcd, -100, 2, 1)
    unoc_pcd = utils.update_points(unoc_pcd, -100, 2, 1)
    
    # Apply frontier transformation
    pcd.transform(utils.inverse_homogeneous_transform(frontier_trans))
    unoc_pcd.transform(utils.inverse_homogeneous_transform(frontier_trans))
    
    # Save processed point cloud
    o3d.io.write_point_cloud("running_occ_.pcd", pcd)

    # Extract local points
    local_points = utils.points_within_distance(0.0, 0.0, np.asarray(pcd.points), 3.2)
    local_points = local_points[local_points[:, 2] < 1.0]
    local_points_unoc = utils.points_within_distance(0.0, 0.0, np.asarray(unoc_pcd.points), 3.2)
    local_points_unoc = local_points_unoc[local_points_unoc[:, 2] < 1.0]
    
    # Create point clouds for visualization
    local_pcd = o3d.geometry.PointCloud()
    local_pcd.points = o3d.utility.Vector3dVector(local_points)
    local_unoc_pcd = o3d.geometry.PointCloud()
    local_unoc_pcd.points = o3d.utility.Vector3dVector(local_points_unoc)

    # Generate pointmaps
    local_octomap_pm = utils.pc_to_pointmap(
        local_points,
        voxel_size=0.1,
        x_y_bounds=[-3.2, 3.2],
        z_bounds=[-1.3, 1.0]
    )
    local_octomap_pm_unon = utils.pc_to_pointmap(
        local_points_unoc,
        voxel_size=0.1,
        x_y_bounds=[-3.2, 3.2],
        z_bounds=[-1.3, 1.0]
    )
    
    print(f"Pointmap shape: {local_octomap_pm.shape}")
    
    return local_octomap_pm, local_octomap_pm_unon

def send_data_to_server(local_octomap_pm, local_octomap_pm_unon):
    """
    Send point cloud data to the diffusion server for processing.
    
    Args:
        local_octomap_pm: Occupied points pointmap
        local_octomap_pm_unon: Unoccupied points pointmap
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Configure print options so arrays don't get truncated
    np.set_printoptions(threshold=np.prod(local_octomap_pm.shape))
    
    # Prepare data to send
    data_to_send = {
        "occ_shape": str(local_octomap_pm.shape),
        "occ_data": np.array2string(local_octomap_pm.flatten()),
        "unocc_shape": str(local_octomap_pm_unon.shape),
        "unocc_data": np.array2string(local_octomap_pm_unon.flatten())
    }

    # Send POST request with JSON data
    response = requests.post(DIFFUSION_ENDPOINT, json=data_to_send)

    # Process the response
    if response.status_code == 200:
        return_data = response.json()
        
        # Parse returned points
        diff_points = np.fromstring(return_data['yourData']["diff_points"][1:-1], dtype=float, sep=" ")
        diff_shape = np.fromstring(return_data['yourData']["point_shape"][1:-1], dtype=int, sep=",")
        print(f"Received point cloud shape: {diff_shape}")
        
        # Reshape and visualize points
        diff_points = diff_points.reshape(diff_shape[0], diff_shape[1])
        print('Successfully received response from server')
        
        # Create and save point cloud
        pcd_inpaint = o3d.geometry.PointCloud()
        pcd_inpaint.points = o3d.utility.Vector3dVector(diff_points)
        o3d.io.write_point_cloud("test_inpainted_pcd.pcd", pcd_inpaint)
        
        # Visualize the point cloud
        coor = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([pcd_inpaint, coor])
        return True
    else:
        print(f'Failed to send data: {response.status_code}')
        return False

def main():
    """Main function to run the client test."""
    # Test basic server connection
    if not test_server_connection():
        print("Aborting due to server connection failure")
        return
        
    # Prepare point cloud data
    local_octomap_pm, local_octomap_pm_unon = prepare_point_cloud_data()
    
    # Send data to server and process results
    send_data_to_server(local_octomap_pm, local_octomap_pm_unon)

if __name__ == "__main__":
    main()
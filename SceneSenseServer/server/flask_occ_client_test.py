from flask import Flask
import requests

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


#get local occ info
import open3d as o3d
from natsort import natsorted
import os
import numpy as np
import utils.utils as utils
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
from diffusers import UNet2DModel
import torch
from diffusers import DDPMScheduler
import torch.nn.functional as F
import os
from natsort import natsorted
import numpy as np
import copy
import open3d as o3d
from tqdm.auto import tqdm
import wandb
import random
from huggingface_hub import login
from diffusers.optimization import get_cosine_schedule_with_warmup
import utils.utils as utils
from scipy.spatial.transform import Rotation
import re
from spconv.pytorch.utils import PointToVoxel
import cv2
from cleanfid import fid
import sys
import shutil

points_path = "data/range_max/running_occ.pcd"
unoc_points = "data/range_max/test_unoc.pcd"
# point_file_names = natsorted(os.listdir(points_path))
pcd = o3d.io.read_point_cloud(points_path)
pcd = utils.update_points(pcd,-1.3,1,2)
unoc_pcd = o3d.io.read_point_cloud(unoc_points)
unoc_pcd = utils.update_points(unoc_pcd,-1.3,1,2)

# pcd = utils.update_points(pcd,0,10,1)

coor = o3d.geometry.TriangleMesh.create_coordinate_frame()

#load the current pose
odom_path = "/hdd/spot_diff_data/odometry/odometry"
odom_file_names = natsorted(os.listdir(odom_path))

pose = np.load(os.path.join(odom_path,odom_file_names[480]))
rotation_obj = Rotation.from_rotvec(pose[3::])
hm_tx_mat = utils.homogeneous_transform(pose[0:3], rotation_obj.as_quat())
#translation to frontier
frontier_trans = utils.homogeneous_transform(np.array([3.0, -2.0, 0]), np.array([0,0,0,1]))
print(hm_tx_mat)
pcd.transform(utils.inverse_homogeneous_transform(hm_tx_mat))
unoc_pcd.transform(utils.inverse_homogeneous_transform(hm_tx_mat))
#clean up the spurious detections for example
pcd = utils.update_points(pcd,-100,2,1)
unoc_pcd = utils.update_points(unoc_pcd,-100,2,1)
pcd.transform(utils.inverse_homogeneous_transform(frontier_trans))
unoc_pcd.transform(utils.inverse_homogeneous_transform(frontier_trans))
# o3d.visualization.draw_geometries([pcd, coor])
o3d.io.write_point_cloud("running_occ_.pcd", pcd)

#get the local info. 
local_points = utils.points_within_distance(0.0,0.0,np.asarray(pcd.points),3.2)
local_points = local_points[local_points[:,2] < 1.0]
local_points_unoc = utils.points_within_distance(0.0,0.0,np.asarray(unoc_pcd.points),3.2)
local_points_unoc = local_points_unoc[local_points_unoc[:,2] < 1.0]
local_pcd = o3d.geometry.PointCloud()
local_pcd.points = o3d.utility.Vector3dVector(local_points)
local_unoc_pcd = o3d.geometry.PointCloud()
local_unoc_pcd.points = o3d.utility.Vector3dVector(local_points_unoc)
# o3d.visualization.draw_geometries([local_pcd, coor])


#generate pointmaps
local_octomap_pm = utils.pc_to_pointmap(local_points, 
                                            voxel_size = 0.1,
                                            x_y_bounds = [-3.2, 3.2],
                                            z_bounds = [-1.3, 1.0])
local_octomap_pm_unon = utils.pc_to_pointmap(local_points_unoc, 
                                        voxel_size = 0.1,
                                        x_y_bounds = [-3.2, 3.2],
                                        z_bounds = [-1.3, 1.0])     

print(local_octomap_pm.shape)
print(local_octomap_pm.flatten())

#configure print options so the arrays dont get truncasted
np.set_printoptions(threshold = np.prod(local_octomap_pm.shape))
# Data to send (as a dictionary that will be converted to JSON)
data_to_send = {
    "occ_shape": str(local_octomap_pm.shape),
    "occ_data": np.array2string(local_octomap_pm.flatten()),
    "unocc_shape": str(local_octomap_pm_unon.shape),
    "unocc_data": np.array2string(local_octomap_pm_unon.flatten())
}

# Send a POST request with JSON data
response = requests.post(post_url, json=data_to_send)

# Check the response from the server
if response.status_code == 200:
    return_data = response.json()
    # print(return_data)
    diff_points = np.fromstring(return_data['yourData']["diff_points"][1:-1], dtype=float, sep = " ") 
    diff_shape = np.fromstring(return_data['yourData']["point_shape"][1:-1], dtype=int, sep = ",") 
    print(diff_shape)
    diff_points = diff_points.reshape(diff_shape[0], diff_shape[1])
    print('Response from server:')
    pcd_inpaint = o3d.geometry.PointCloud()
    pcd_inpaint.points = o3d.utility.Vector3dVector(diff_points)
    o3d.io.write_point_cloud("test_inpainted_pcd.pcd", pcd_inpaint)
    o3d.visualization.draw_geometries([pcd_inpaint, coor])
else:
    print('Failed to send data:', response.status_code)
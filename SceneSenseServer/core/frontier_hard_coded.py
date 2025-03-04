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
import time

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
# frontier_trans = utils.homogeneous_transform(np.array([3.0, -2.0, 0]), np.array([0,0,0,1]))

pcd.transform(utils.inverse_homogeneous_transform(hm_tx_mat))
unoc_pcd.transform(utils.inverse_homogeneous_transform(hm_tx_mat))
#clean up the spurious detections for example
pcd = utils.update_points(pcd,-100,2,1)
unoc_pcd = utils.update_points(unoc_pcd,-100,2,1)
# pcd.transform(utils.inverse_homogeneous_transform(frontier_trans))
# unoc_pcd.transform(utils.inverse_homogeneous_transform(frontier_trans))
# o3d.visualization.draw_geometries([pcd, coor])

#get the local info. 
local_points = utils.points_within_distance(0.0,0.0,np.asarray(pcd.points),3.3)
#update for new model
local_points = local_points[local_points[:,2] < 0.9]
local_points = local_points[local_points[:,2] > -1.0]
local_unocc_points = utils.points_within_distance(0.0,0.0,np.asarray(unoc_pcd.points),3.3)
local_unocc_points = local_unocc_points[local_unocc_points[:,2] < 0.9]
local_unocc_points = local_unocc_points[local_unocc_points[:,2] > -1.0]
# o3d.visualization.draw_geometries([local_pcd, coor])

local_pcd = o3d.geometry.PointCloud()
local_pcd.points = o3d.utility.Vector3dVector(local_points)
# o3d.visualization.draw_geometries([local_pcd])
#do the diffusion
torch_device = "cuda"
#get the models
model = UNet2DModel.from_pretrained("alre5639/frontier_diff_try_3").to(torch_device)
# conditioning_model = get_model()
sample_noise_start = torch.randn(1,13,64,64).to(torch_device)
print("Unet output shape:", model(sample_noise_start, timestep=1.0).sample.shape)


#generate pointmaps
local_occ_pm = utils.pc_to_pointmap(local_points, 
                                        voxel_size = 0.15,
                                        x_y_bounds = [-3.3, 3.3],
                                        z_bounds = [-1.0, 0.9])
local_unocc_pm = utils.pc_to_pointmap(local_unocc_points, 
                                        voxel_size = 0.15,
                                        x_y_bounds = [-3.3, 3.3],
                                        z_bounds = [-1.0, 0.9])
#need to add the zero padding
pad_height = (64 - 44) // 2  # Padding for height
pad_width = (64 - 44) // 2   # Padding for width
local_occ_pm = np.pad(local_occ_pm, pad_width=((0, 0), (pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
local_unocc_pm = np.pad(local_unocc_pm, pad_width=((0, 0), (pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
# check pointmaps are correct
for i, img in enumerate(np.asarray(local_occ_pm)):
    #normalize the outputs to 255 in each pixel
    output = copy.deepcopy(img) * 255
    #dupicate it to be an image
    output = np.repeat(output[:, :, np.newaxis], 3, axis=2)

    #save it as a cv2 image
    cv2.imwrite("data/range_max/input_images/" + str(i) + ".png", output )

# check pointmaps are correct
for i, img in enumerate(np.asarray(local_unocc_pm)):
    #normalize the outputs to 255 in each pixel
    output = copy.deepcopy(img) * 255
    #dupicate it to be an image
    output = np.repeat(output[:, :, np.newaxis], 3, axis=2)

    #save it as a cv2 image
    cv2.imwrite("data/range_max/unoc_images/" + str(i) + ".png", output )
o3d.visualization.draw_geometries([local_pcd, coor])


noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
# noise_scheduler = noise_scheduler.to(torch_device)
start_time = time.time()
inpained_pm = utils.efficient_inpainting_pointmaps_w_freespace(model,
                                                noise_scheduler,
                                                64,
                                                local_occ_pm,
                                                local_unocc_pm,
                                                torch_device = "cuda",
                                                denoising_steps = int(30),
                                                guidance_scale = int(0),
                                                sample_batch_size = 1)
print("diff time: ", time.time() - start_time)
#remove padding
print(inpained_pm.shape)
inpained_pm = inpained_pm[:, :, pad_height:-pad_height, pad_width:-pad_width]

# check pointmaps are correct
for i, img in enumerate(np.asarray(inpained_pm[0].cpu())):
    #normalize the outputs to 255 in each pixel
    output = copy.deepcopy(img) * 255
    #dupicate it to be an image
    output = np.repeat(output[:, :, np.newaxis], 3, axis=2)

    #save it as a cv2 image
    cv2.imwrite("data/range_max/diff_out/" + str(i) + ".png", output )

inpained_points = utils.pointmap_to_pc_fast(inpained_pm[0],
                                                voxel_size = 0.15,
                                                x_y_bounds = [-3.3, 3.3],
                                                z_bounds = [-1.0, 0.9])
pcd_inpaint = o3d.geometry.PointCloud()
pcd_inpaint.points = o3d.utility.Vector3dVector(inpained_points)
# o3d.visualization.draw_geometries([local_pcd, coor])
o3d.io.write_point_cloud("test_inpainted_pcd.pcd", pcd_inpaint)
o3d.visualization.draw_geometries([pcd_inpaint, coor])
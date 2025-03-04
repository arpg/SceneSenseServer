from flask import Flask, request, jsonify
from urllib.parse import urlparse
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
import json
import time

app = Flask(__name__)

#updated this for tailscale, this is the tailscale ip for arpg desktop
host_url = "http://100.96.130.45:5000"
# host_url = "http://172.31.20.53:5000" #for direct ethernet 


parsed_url = urlparse(host_url)

host = parsed_url.hostname
port = parsed_url.port
print("Host", host)

torch_device = "cuda"
#get the models
model = UNet2DModel.from_pretrained("alre5639/frontier_diff_try_3").to(torch_device)#, revision = "b063adc01ea748b7a4dbfb7e180eedf741aef536").to(torch_device)
# conditioning_model = get_model()
sample_noise_start = torch.randn(1,13,64,64).to(torch_device)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# Define a route
@app.route('/')
def hello():
    return 'connected to diffusion server'

#define the route to receive the occupancy JSON
@app.route('/diffusion', methods=['POST'])
def handle_data():
    # Check if the request contains JSON data
    if request.is_json:
        # Get the JSON data
        data = request.get_json()
        
        #save the json
        with open('frontiers_recieved_data.json', 'w') as file:
            json.dump(data, file, indent=4)
        # Process or respond to the data

        #check the time its in microsecnds
        send_time = float(data["time"])
        # Get the current time in seconds since the epoch
        current_time_seconds = time.time()

        # Convert seconds to microseconds
        current_time_microseconds = int(current_time_seconds * 1_000_000)
        print("\nTx time to recieve is: ", current_time_microseconds - send_time)
        

        # occ_shape = np.fromstring(data["occ_shape"][1:-1], dtype=int, sep = ",")
        occ_data =np.fromstring(data["occ_data"][1:-2], dtype=float, sep = " ") 
        occ_data = occ_data.reshape(-1,3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(occ_data)
        # load unocc data
        unocc_data =np.fromstring(data["unocc_data"][1:-2], dtype=float, sep = " ") 
        unocc_data = unocc_data.reshape(-1,3)
        unocc_pcd = o3d.geometry.PointCloud()
        unocc_pcd.points = o3d.utility.Vector3dVector(unocc_data)

        coor = o3d.geometry.TriangleMesh.create_coordinate_frame()
        
        #diffusion point
        # target_point =  np.fromstring(data["odom"], dtype=float, sep = " ") 
        # path_pcd = o3d.geometry.PointCloud()
        # path_pcd.points = o3d.utility.Vector3dVector(target_point)
        # colors = np.zeros((len(np.asarray(path_pcd.points)), 3))
        # colors[:,1] = 1
        # path_pcd.colors = o3d.utility.Vector3dVector(colors)

        # o3d.visualization.draw_geometries([pcd, unocc_pcd, coor])

        print(data["odom"])
        odom_data = np.fromstring(data["odom"][1:-2], dtype=float, sep = " ") 
        # print(odom_data)
        odom_translation = odom_data[0:3]
        odom_rotataton = Rotation.from_quat(odom_data[3::])
        # print(odom_data)
        # transform pointcloud
        # print(odom_rotataton)
        print(odom_data, odom_data.shape)

        hx_tx = utils.homogeneous_transform(odom_translation,odom_data[3::])
        pcd.transform(utils.inverse_homogeneous_transform(hx_tx))
        unocc_pcd.transform(utils.inverse_homogeneous_transform(hx_tx))
        local_points = utils.points_within_distance(0.0,0.0,np.asarray(pcd.points),3.3)
        #update for new model
        local_points = local_points[local_points[:,2] < 0.9]
        local_points = local_points[local_points[:,2] > -1.0]
        local_unocc_points = utils.points_within_distance(0.0,0.0,np.asarray(unocc_pcd.points),3.3)
        local_unocc_points = local_unocc_points[local_unocc_points[:,2] < 0.9]
        local_unocc_points = local_unocc_points[local_unocc_points[:,2] > -1.0]
        local_pcd = o3d.geometry.PointCloud()
        local_pcd.points = o3d.utility.Vector3dVector(local_points)
        local_unoc_pcd = o3d.geometry.PointCloud()
        local_unoc_pcd.points = o3d.utility.Vector3dVector(local_unocc_points)
        colors = np.zeros((len(np.asarray(local_unoc_pcd.points)), 3))
        colors[:,0] = 1
        local_unoc_pcd.colors = o3d.utility.Vector3dVector(colors)

        # o3d.visualization.draw_geometries([local_pcd,local_unoc_pcd, coor])
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
        # occ_data = occ_data.reshape(13,64,64)
        # unocc_data =np.fromstring(data["unocc_data"][1:-2], dtype=float, sep = " ") 
        # unocc_data = unocc_data.reshape(13,64,64)
        to_diff_time = int(time.time() * 1_000_000)
        print("up to diffusion: ", to_diff_time - current_time_microseconds, "")
        
        inpained_pm = utils.efficient_inpainting_pointmaps_w_freespace(model,
                                                noise_scheduler,
                                                64,
                                                local_occ_pm,
                                                local_unocc_pm,
                                                torch_device = "cuda",
                                                denoising_steps = int(30),
                                                guidance_scale = int(0),
                                                sample_batch_size = 1)
        #remove padding
        print(inpained_pm.shape)
        inpained_pm = inpained_pm[:, :, pad_height:-pad_height, pad_width:-pad_width]
        print(inpained_pm.shape)
        diff_time = int(time.time() * 1_000_000)
        print("diffuison time: ", diff_time - to_diff_time)       
        #trying to make the pointmap to point cloud faster
        #the above util call takes 1.5 seconds
        inpained_points = utils.pointmap_to_pc_fast(inpained_pm[0],
                                                voxel_size = 0.15,
                                                x_y_bounds = [-3.3, 3.3],
                                                z_bounds = [-1.0, 0.9])
        inpained_points_unoc = utils.pointmap_to_pc_fast_unoc(inpained_pm[0],
                                                voxel_size = 0.15,
                                                x_y_bounds = [-3.3, 3.3],
                                                z_bounds = [-1.0, 0.9])
        print("pointmap to pointcloud time: ", int(time.time() * 1_000_000) - diff_time, "\n")
        # need to add the padding
        
        diff_pcd = o3d.geometry.PointCloud()
        diff_pcd.points = o3d.utility.Vector3dVector(inpained_points)
        diff_pcd.transform(hx_tx)
        transformed_points = np.asarray(diff_pcd.points)

        #for unoccupied points
        diff_pcd_unoc = o3d.geometry.PointCloud()
        diff_pcd_unoc.points = o3d.utility.Vector3dVector(inpained_points_unoc)
        diff_pcd_unoc.transform(hx_tx)
        transformed_points_unoc = np.asarray(diff_pcd_unoc.points)
        
        # o3d.visualization.draw_geometries([diff_pcd,coor])

        # diff_pcd = o3d.geometry.PointCloud()
        # diff_pcd.points = o3d.utility.Vector3dVector(inpained_points_fast)
        # diff_pcd.transform(hx_tx)
        # transformed_points = np.asarray(diff_pcd.points)

        # o3d.visualization.draw_geometries([diff_pcd,coor])

        flat_points = transformed_points.flatten()
        return_string = "["
        for point in  flat_points:
            return_string += f"{point:.2f}"
            return_string += " "
        return_string += "]"
        # need to replace double spaces
        return_string = re.sub(r' +', ' ', return_string)

        #for unoccupied points
        flat_points_unoc = transformed_points_unoc.flatten()
        return_string_unoc = "["
        for point in  flat_points_unoc:
            return_string_unoc += f"{point:.2f}"
            return_string_unoc += " "
        return_string_unoc += "]"
        # need to replace double spaces
        return_string_unoc = re.sub(r' +', ' ', return_string_unoc)
        
        # np.set_printoptions(threshold = np.prod(inpained_points.shape))
        np.set_printoptions(precision = 30, threshold = sys.maxsize)
        # np.set_printoptions(precision = 30)
        # np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        #this only sometimes works for some reason

        data_to_return={
            "occ_points": return_string,
            "unoc_points": return_string_unoc
        }
        #save the json
        with open('frontier_data_to_send.json', 'w') as file:
            json.dump(data_to_return, file, indent=4)
        to_tx_time = int(time.time() * 1_000_000)
        print("post_diffusion time: ", to_tx_time - diff_time, "\n")
        return jsonify({"message": "Data received", "yourData": data_to_return}), 200 #the 200 here is the return status code (arbitarty)
    else:
        return jsonify({"error": "Request must be JSON"}), 400 

app.run(debug=True, host=host, port=port, use_reloader=False)
"""
Flask server implementation for diffusion-based point cloud completion.

This module provides a Flask server that receives point cloud data in JSON format,
applies a diffusion model for completion, and returns the completed point cloud.
"""

import copy
import json
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from urllib.parse import urlparse

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import wandb
from cleanfid import fid
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from flask import Flask, jsonify, request
from huggingface_hub import login
from natsort import natsorted
from scipy.spatial.transform import Rotation
from spconv.pytorch.utils import PointToVoxel
from tqdm.auto import tqdm

from SceneSenseServer.utils import utils

app = Flask(__name__)

# Server configuration
host_url = "http://0.0.0.0:5000"
parsed_url = urlparse(host_url)
host = parsed_url.hostname
port = parsed_url.port
print("Host", host)

# Set up the diffusion model
torch_device = "cuda"
model = UNet2DModel.from_pretrained("alre5639/frontier_diff_no_cond_try_2").to(torch_device)
sample_noise_start = torch.randn(1, 23, 64, 64).to(torch_device)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)


@app.route("/")
def hello():
    """
    Root endpoint to check if the server is running.
    
    Returns:
        str: A simple message indicating the server is connected.
    """
    return "connected to diffusion server"


@app.route("/diffusion", methods=["POST"])
def handle_data():
    """
    Handle POST requests with point cloud data for diffusion-based completion.
    
    Expects a JSON payload containing:
    - occ_data: Occupied points data as a flattened string array
    - unocc_data: Unoccupied points data as a flattened string array
    
    Returns:
        tuple: JSON response with the completed point cloud data and HTTP status code
    """
    # Check if the request contains JSON data
    if request.is_json:
        # Get the JSON data
        data = request.get_json()

        # Save the received JSON for debugging
        with open("recieved_data.json", "w") as file:
            json.dump(data, file, indent=4)
            
        # Convert data strings to numpy arrays and reshape
        occ_data = np.fromstring(data["occ_data"][1:-2], dtype=float, sep=" ")
        occ_data = occ_data.reshape(23, 64, 64)
        unocc_data = np.fromstring(data["unocc_data"][1:-2], dtype=float, sep=" ")
        unocc_data = unocc_data.reshape(23, 64, 64)

        # Apply diffusion model for point cloud completion
        inpainted_pm = utils.efficient_inpainting_pointmaps_w_freespace(
            model,
            noise_scheduler,
            64,
            occ_data,
            unocc_data,
            torch_device="cuda",
            denoising_steps=int(30),
            guidance_scale=int(0),
            sample_batch_size=1,
        )

        # Convert the pointmap back to a 3D point cloud
        inpainted_points = utils.pointmap_to_pc(
            inpainted_pm[0], 
            voxel_size=0.1, 
            x_y_bounds=[-3.2, 3.2], 
            z_bounds=[-1.3, 1.0]
        )
        
        # Prepare the data for response
        np.set_printoptions(threshold=np.prod(inpainted_points.shape))
        data_to_return = {
            "point_shape": str(inpainted_points.shape),
            "diff_points": np.array2string(inpainted_points.flatten()),
        }
        
        return jsonify({"message": "Data received", "yourData": data_to_return}), 200
    else:
        return jsonify({"error": "Request must be JSON"}), 400


if __name__ == "__main__":
    app.run(debug=True, host=host, port=port, use_reloader=False)

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

host_url = "http://0.0.0.0:5000"

parsed_url = urlparse(host_url)

host = parsed_url.hostname
port = parsed_url.port
print("Host", host)

torch_device = "cuda"
# get the models
model = UNet2DModel.from_pretrained("alre5639/frontier_diff_no_cond_try_2").to(
    torch_device
)  # , revision = "b063adc01ea748b7a4dbfb7e180eedf741aef536").to(torch_device)
# conditioning_model = get_model()
sample_noise_start = torch.randn(1, 23, 64, 64).to(torch_device)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)


# Define a route
@app.route("/")
def hello():
    return "connected to diffusion server"


# define the route to receive the occupancy JSON
@app.route("/diffusion", methods=["POST"])
def handle_data():
    # Check if the request contains JSON data
    if request.is_json:
        # Get the JSON data
        data = request.get_json()

        # save the json
        with open("recieved_data.json", "w") as file:
            json.dump(data, file, indent=4)
        # Process or respond to the data
        # occ_shape = np.fromstring(data["occ_shape"][1:-1], dtype=int, sep = ",")
        occ_data = np.fromstring(data["occ_data"][1:-2], dtype=float, sep=" ")
        occ_data = occ_data.reshape(23, 64, 64)
        unocc_data = np.fromstring(data["unocc_data"][1:-2], dtype=float, sep=" ")
        unocc_data = unocc_data.reshape(23, 64, 64)

        inpained_pm = utils.efficient_inpainting_pointmaps_w_freespace(
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

        inpained_points = utils.pointmap_to_pc(
            inpained_pm[0], voxel_size=0.1, x_y_bounds=[-3.2, 3.2], z_bounds=[-1.3, 1.0]
        )
        np.set_printoptions(threshold=np.prod(inpained_points.shape))
        data_to_return = {
            "point_shape": str(inpained_points.shape),
            "diff_points": np.array2string(inpained_points.flatten()),
        }
        return (
            jsonify({"message": "Data received", "yourData": data_to_return}),
            200,
        )  # the 200 here is the return status code (arbitarty)
    else:
        return jsonify({"error": "Request must be JSON"}), 400


app.run(debug=True, host=host, port=port, use_reloader=False)

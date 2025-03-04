from dataclasses import dataclass
from diffusers import UNet2DModel
import torch
from diffusers import DDPMScheduler
import torch.nn.functional as F
import os
from natsort import natsorted
import numpy as np
import copy
# import open3d as o3d
from tqdm.auto import tqdm
import wandb
import random
from huggingface_hub import login
from diffusers.optimization import get_cosine_schedule_with_warmup
import re
import utils.utils as utils
import time

login(token="Your Huggingface Token")
wandb.init(
    # set the wandb project where this run will be logged
    project="scene_diffusion",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

@dataclass
class TrainingConfig:
    image_size = 44  # the generated image resolution
    train_batch_size = 32
    eval_batch_size = 32  # how many images to sample during evaluation
    num_epochs = 250
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 20
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "baseline_road_diffusion"  # the model name locally and on the HF Hub

    push_to_hub = True  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

torch_device = "cuda"
config = TrainingConfig()
#need to pad data to 64x64
model = UNet2DModel(
    sample_size=64,  # the target image resolution
    in_channels=13,  # the number of input channels, 3 for RGB images
    out_channels=13,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
    ),
).to(torch_device)

# make sure all the data moves through the network correctly
sample_noise_start = torch.randn(1,13,64, 64).to(torch_device)

shape_out = model(sample_noise_start, timestep=0.0).sample.shape
print("Unet output shape:", shape_out)


# ########################
# #get gt data
# #########################3
gt_dir = "/home/cc/updated_gt_maps"
gt_files = natsorted(os.listdir(gt_dir))
gt_full_path = []
#get the files for each dir
for idx, gt_file in enumerate(gt_files):
    gt_full_path.append(os.path.join(gt_dir, gt_file))
print(gt_full_path)
print(len(gt_full_path))
# # ##########################################################
# # #Code for viewing test input and gt pointcloud
# # ########################################################
# # test_pm = np.load(gt_dir + aligned_gt_files[0])
# # gt_points = utils.pointmap_to_pc(test_pm, voxel_size  = 0.1, x_y_bounds = [-2, 2], z_bounds = [-1.4, 0.9])
# # conditioning_data = np.load(cond_dir + aligned_cond_files[0])
# # print(conditioning_data.shape)

# # pcd_local = o3d.geometry.PointCloud()
# # pcd_conditioning = o3d.geometry.PointCloud()
# # pcd_local.points = o3d.utility.Vector3dVector(gt_points)
# # pcd_conditioning.points = o3d.utility.Vector3dVector(conditioning_data[:,0:3])
# # o3d.visualization.draw_geometries([pcd_local,pcd_conditioning])




###################################
# try and load the whole array into memory
##############################################
# data_array = np.empty((0,12,44,44),float)
# for my_path in gt_full_path:
#     data = np.load(my_path)
#     data = data[None,:,:,:]
#     data_array = np.append(data_array,data, axis = 0)
#     print(data_array.shape)

# # ##############################################
# #create data loaders
# #############################################
# shuffle arrays:
np.random.seed(1)
np.random.shuffle(gt_full_path)
# # print(gt_data.shape)
# # print(len(cond_files))
gt_dataloader = torch.utils.data.DataLoader(gt_full_path, batch_size=config.train_batch_size, shuffle=False)
# conditioning_dataloader = torch.utils.data.DataLoader(aligned_cond_files, batch_size=config.train_batch_size, shuffle=False)


# # ################################################
# # # setup training stuff
# # ################################################
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
bs = sample_noise_start.shape[0]
timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=sample_noise_start.device).long()

global_step = 0

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(gt_dataloader) * config.num_epochs),
)

#make sure everything is on the GPU
model = model.to(torch_device)

for epoch in range(config.num_epochs):
    progress_bar = tqdm(total=len(gt_dataloader))
    progress_bar.set_description(f"Epoch {epoch}")

    for step, batch in enumerate(gt_dataloader):
        #####################################################
        #load the gt data
        ####################################################
        #initalize gt data cube:
        gt_data = np.load(batch[0])
        #add batch dim
        gt_data = gt_data[None,:,:,:]

        for gt_file in batch[1:]:
            single_gt_data = np.load(gt_file)
            #add batch dim
            single_gt_data = single_gt_data[None,:,:,:]
            #append to data cube
            gt_data = np.append(gt_data, single_gt_data, axis = 0)
        gt_data = gt_data.astype(np.single)
        #gt_data to tensor:
        gt_data = torch.tensor(gt_data).to(torch_device)
        #######################3
        # turn gt data into the images
        #here what it does in the old scripts
        clean_images = gt_data
        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
        noisy_images = noisy_images.to(torch_device)
 
        # print(noisy_images.dtype)
        # print(post_model_conditioning_batch.dtype)
        # print(noisy_images.shape)
        # print(post_model_conditioning_batch.shape)
        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()
        # torch.nn.utils.clip_grad_norm(list(model.parameters()) + list(conditioning_model.parameters()),options['clip_gradient_norm'])

        #NEED TO ADD THIS 
        # accelerator.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress_bar.update(1)
        wandb.log({"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step, "epoch": epoch})

        global_step += 1

        # should add validation step here, FID calculation would be good

    if epoch%25 == 0:
        print("\nPushing to Hub\n")
        try:
            model.push_to_hub("frontier_diff_try_3")
        except:
            print("push failed")
# #     # repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
# #     # conditioning_model.push_to_hub("diff_pointnet")
# #     # repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)






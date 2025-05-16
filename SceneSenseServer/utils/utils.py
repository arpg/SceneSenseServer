"""
Core utility functions for point cloud processing and diffusion-based scene completion.

This module contains various utility functions for working with point clouds,
coordinate transformations, pointmap conversions, and applying diffusion models
for scene completion tasks.
"""

import copy
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm


def inverse_homogeneous_transform(matrix):
    """
    Calculate the inverse of a homogeneous transformation matrix.
    
    Args:
        matrix (np.ndarray): A 4x4 homogeneous transformation matrix.
        
    Returns:
        np.ndarray: The inverse 4x4 homogeneous transformation matrix.
        
    Raises:
        ValueError: If the input matrix is not 4x4.
    """
    if matrix.shape != (4, 4):
        raise ValueError("Input matrix must be a 4x4 numpy array")
    
    # Extract the rotation and translation components
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    
    # Compute the inverse of the rotation matrix
    inverse_rotation = np.transpose(rotation)
    
    # Compute the inverse translation
    inverse_translation = -np.dot(inverse_rotation, translation)
    
    # Construct the inverse homogeneous transform matrix
    inverse_matrix = np.eye(4)
    inverse_matrix[:3, :3] = inverse_rotation
    inverse_matrix[:3, 3] = inverse_translation
    
    return inverse_matrix

def homogeneous_transform(translation, rotation):
    """
    Generate a homogeneous transformation matrix from a translation vector
    and a quaternion rotation.

    Args:
        translation (np.ndarray): 1D array of length 3 representing translation along x, y, and z axes.
        rotation (np.ndarray): 1D array of length 4 representing a quaternion rotation.

    Returns:
        np.ndarray: 4x4 homogeneous transformation matrix.
        
    Raises:
        ValueError: If translation vector or rotation quaternion have incorrect dimensions.
    """
    # Ensure that the input vectors have the correct dimensions
    translation = np.array(translation, dtype=float)
    rotation = np.array(rotation, dtype=float)

    if translation.shape != (3,) or rotation.shape != (4,):
        raise ValueError("Translation vector must be of length 3, and rotation quaternion must be of length 4.")

    # Normalize the quaternion to ensure it is a unit quaternion
    rotation /= np.linalg.norm(rotation)

    # Create a rotation matrix from the quaternion using scipy's Rotation class
    rotation_matrix = Rotation.from_quat(rotation).as_matrix()

    # Create a 4x4 homogeneous transformation matrix
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:3, 3] = translation

    return homogeneous_matrix

def update_points(pcd, min_val, max_val, axis):
    """
    Filter points in a point cloud based on min/max bounds along a specified axis.
    
    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        min_val (float): Minimum value along the specified axis.
        max_val (float): Maximum value along the specified axis.
        axis (int): The axis to filter (0=x, 1=y, 2=z).
        
    Returns:
        o3d.geometry.PointCloud: Filtered point cloud.
    """
    points = np.asarray(pcd.points)
    points = points[points[:, axis] < max_val]
    points = points[points[:, axis] > min_val]
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def set_rgb(pcd, color_idx=None):
    """
    Set RGB colors for a point cloud.
    
    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        color_idx (int, optional): Index of the color channel to set to 1 (R=0, G=1, B=2).
            If None, all colors are set to 0.
            
    Returns:
        o3d.geometry.PointCloud: Point cloud with updated colors.
    """
    colors = np.zeros((len(np.asarray(pcd.points)), 3))
    if color_idx is not None:
        colors[:, color_idx] = 1.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def points_within_distance(x, y, points, distance):
    """
    Find all 3D points within a specified distance from a given (x, y) location.

    Args:
        x (float): The x coordinate of the reference location.
        y (float): The y coordinate of the reference location.
        points (np.ndarray): Array of shape (num_points, 3) representing 3D points.
        distance (float): The maximum distance for points to be considered within.

    Returns:
        np.ndarray: Array of points within the specified distance.
    """
    # Extract x, y coordinates from the 3D points
    xy_coordinates = points[:, [0, 1]]

    # Calculate the Euclidean distance from the given location to all points
    distances = np.linalg.norm(xy_coordinates - np.array([x, y]), axis=1)

    # Find indices of points within the specified distance
    within_distance_indices = np.where(distances <= distance)[0]

    # Extract points within the distance
    points_within = points[within_distance_indices]

    return points_within

def pc_to_pointmap(pointcloud, voxel_size=0.1, x_y_bounds=[-1.5, 1.5], z_bounds=[-1.4, 0.9]):
    """
    Convert a point cloud to a 3D pointmap representation.
    
    Args:
        pointcloud (np.ndarray): Array of shape (num_points, 3) representing 3D points.
        voxel_size (float): Size of each voxel in the pointmap.
        x_y_bounds (list): Minimum and maximum values for x and y dimensions.
        z_bounds (list): Minimum and maximum values for z dimension.
        
    Returns:
        np.ndarray: 3D pointmap representation with shape (z_width, x_y_width, x_y_width).
    """
    # Calculate dimensions of the pointmap
    x_y_width = round((x_y_bounds[1] - x_y_bounds[0]) / voxel_size)
    z_width = round((z_bounds[1] - z_bounds[0]) / voxel_size)
    
    # Create a copy of the pointcloud to avoid modifying the original
    pointcloud = pointcloud.copy()
    
    # Shift points to start at origin (no negative numbers since pointmap indices are positive)
    pointcloud[:, 0] = pointcloud[:, 0] + abs(x_y_bounds[0])
    pointcloud[:, 1] = pointcloud[:, 1] + abs(x_y_bounds[0])
    pointcloud[:, 2] = pointcloud[:, 2] + abs(z_bounds[0])
    
    # Create empty pointmap
    point_map = np.zeros((z_width, x_y_width, x_y_width), dtype=float)
    
    # Compute which voxels to fill in the pointmap
    prec_vox_X = pointcloud[:, 0] / (x_y_bounds[1] - x_y_bounds[0])
    prec_vox_Y = pointcloud[:, 1] / (x_y_bounds[1] - x_y_bounds[0])
    prec_vox_Z = pointcloud[:, 2] / (z_bounds[1] - z_bounds[0])
    
    # For each point, fill a point in the pointmap
    for idx, val in enumerate(prec_vox_X):
        # Bound the indices to be within the pointmap dimensions
        z_idx = min(math.floor(prec_vox_Z[idx] * z_width), z_width - 1)
        x_idx = min(math.floor(prec_vox_X[idx] * x_y_width), x_y_width - 1)
        y_idx = min(math.floor(prec_vox_Y[idx] * x_y_width), x_y_width - 1)
        
        # Skip if indices are negative
        if z_idx < 0 or x_idx < 0 or y_idx < 0:
            continue
            
        point_map[z_idx, x_idx, y_idx] = 1.0

    return point_map

def pointmap_to_pc(pointmap, voxel_size=0.1, x_y_bounds=[-1.5, 1.5], z_bounds=[-1.4, 0.9], prediction_thresh=0.8):
    """
    Convert a pointmap back to a point cloud.
    
    Args:
        pointmap (np.ndarray): 3D pointmap representation.
        voxel_size (float): Size of each voxel in the pointmap.
        x_y_bounds (list): Minimum and maximum values for x and y dimensions.
        z_bounds (list): Minimum and maximum values for z dimension.
        prediction_thresh (float): Threshold for considering a voxel as occupied.
        
    Returns:
        np.ndarray: Array of shape (num_points, 3) representing 3D points.
    """
    # Setup empty pointcloud
    arr = np.empty((0, 3), np.single)
    
    # Iterate through the pointmap
    for z_idx, z_val in enumerate(pointmap):
        for x_idx, x_val in enumerate(z_val):
            for y_idx, y_val in enumerate(x_val):
                if y_val > prediction_thresh:
                    # Add a point at the center of the voxel
                    point = np.array([[
                        x_idx * voxel_size + voxel_size / 2,
                        y_idx * voxel_size + voxel_size / 2,
                        z_idx * voxel_size + voxel_size / 2
                    ]])
                    arr = np.append(arr, point, axis=0)

    # Shift points back to original coordinate system
    arr[:, 0] = arr[:, 0] - abs(x_y_bounds[0])
    arr[:, 1] = arr[:, 1] - abs(x_y_bounds[0])
    arr[:, 2] = arr[:, 2] - abs(z_bounds[0])

    return arr

def pointmap_to_pc_fast(pointmap, voxel_size=0.1, x_y_bounds=[-1.5, 1.5], z_bounds=[-1.4, 0.9], prediction_thresh=0.8):
    """
    Faster version of pointmap_to_pc using NumPy operations.
    
    Args:
        pointmap (torch.Tensor or np.ndarray): 3D pointmap representation.
        voxel_size (float): Size of each voxel in the pointmap.
        x_y_bounds (list): Minimum and maximum values for x and y dimensions.
        z_bounds (list): Minimum and maximum values for z dimension.
        prediction_thresh (float): Threshold for considering a voxel as occupied.
        
    Returns:
        np.ndarray: Array of shape (num_points, 3) representing occupied points.
    """
    # Move to CPU if it's a torch tensor
    if isinstance(pointmap, torch.Tensor):
        local_pm = pointmap.cpu().numpy()
    else:
        local_pm = pointmap
    
    # Get indices where there are predicted points
    point_idx = np.where(local_pm > prediction_thresh)
    
    # Create points array using the indices
    points_arr = np.array([[point_idx[1]], [point_idx[2]], [point_idx[0]]], dtype=float)
    points_arr = np.transpose(points_arr, (1, 2, 0))[0]
    
    # Convert indices to coordinates
    points_arr[:, 0] = points_arr[:, 0] * voxel_size + voxel_size / 2
    points_arr[:, 1] = points_arr[:, 1] * voxel_size + voxel_size / 2
    points_arr[:, 2] = points_arr[:, 2] * voxel_size + voxel_size / 2

    # Shift points back to original coordinate system
    points_arr[:, 0] = points_arr[:, 0] - abs(x_y_bounds[0])
    points_arr[:, 1] = points_arr[:, 1] - abs(x_y_bounds[0])
    points_arr[:, 2] = points_arr[:, 2] - abs(z_bounds[0])

    return points_arr

def pointmap_to_pc_fast_unoc(pointmap, voxel_size=0.1, x_y_bounds=[-1.5, 1.5], z_bounds=[-1.4, 0.9], prediction_thresh=0.8):
    """
    Faster version of pointmap_to_pc for unoccupied space (values below threshold).
    
    Args:
        pointmap (torch.Tensor or np.ndarray): 3D pointmap representation.
        voxel_size (float): Size of each voxel in the pointmap.
        x_y_bounds (list): Minimum and maximum values for x and y dimensions.
        z_bounds (list): Minimum and maximum values for z dimension.
        prediction_thresh (float): Threshold for considering a voxel as unoccupied.
        
    Returns:
        np.ndarray: Array of shape (num_points, 3) representing unoccupied points.
    """
    # Move to CPU if it's a torch tensor
    if isinstance(pointmap, torch.Tensor):
        local_pm = pointmap.cpu().numpy()
    else:
        local_pm = pointmap
        
    # Get indices where there are unoccupied points (below threshold)
    point_idx = np.where(local_pm < prediction_thresh)
    
    # Create points array using the indices
    points_arr = np.array([[point_idx[1]], [point_idx[2]], [point_idx[0]]], dtype=float)
    points_arr = np.transpose(points_arr, (1, 2, 0))[0]
    
    # Convert indices to coordinates
    points_arr[:, 0] = points_arr[:, 0] * voxel_size + voxel_size / 2
    points_arr[:, 1] = points_arr[:, 1] * voxel_size + voxel_size / 2
    points_arr[:, 2] = points_arr[:, 2] * voxel_size + voxel_size / 2

    # Shift points back to original coordinate system
    points_arr[:, 0] = points_arr[:, 0] - abs(x_y_bounds[0])
    points_arr[:, 1] = points_arr[:, 1] - abs(x_y_bounds[0])
    points_arr[:, 2] = points_arr[:, 2] - abs(z_bounds[0])

    return points_arr

def inpainting_pointmaps_w_freespace(model, noise_scheduler, width, inpainting_target, inpainting_unocc, torch_device = "cpu", denoising_steps = 30, guidance_scale = 3, sample_batch_size = 1):
    #set up initialize noise scheudler and noise to be operated on
    noise_scheduler.set_timesteps(denoising_steps, device = torch_device)
    noise = torch.randn(inpainting_target.shape)
    #load the inpainting target
    voxel_grid = torch.tensor(inpainting_target, dtype=torch.float)
    #get the coordinates of the occupied voxels
    input_coordinate = np.where(voxel_grid > 0.9)
    print(input_coordinate)
    #get the coordinates of the unoccupied voxels
    unnoc_grid = torch.tensor(inpainting_unocc, dtype = torch.float)
    unnoc_coordinate = np.where(unnoc_grid > 0.9)
    # print("shape: ", input_coordinate.shape)

    #generate noise to be diffused
    generator = torch.manual_seed(0)  # Seed generator to create the inital latent noise

    #set sample conditioning, concating uncondtioned noise so we only need to do one forward pass

    #generate progress bar
    #perform diffusion
    # print(noise_scheduler.timesteps)
    generator = torch.manual_seed(0)  
    latents = torch.randn((sample_batch_size, model.config.in_channels, width, width),generator=generator)
    latents = latents.to(torch_device)
    latents = latents * noise_scheduler.init_noise_sigma
    for t in tqdm(noise_scheduler.timesteps):
        #get the noisy scan points
        #this adds noise to voxel grid which is our conditioning targets
        noisy_images = noise_scheduler.add_noise(voxel_grid, noise, timesteps = torch.tensor([t.item()]))
        #WE ACTUALLY NEED TO TURN THIS TO Zeros so we do 1 -
        #I think this 1- operation might mess it up
        noisy_unoc_images = noise_scheduler.add_noise(1 - unnoc_grid, noise, timesteps = torch.tensor([t.item()]))
        # print(latents.shape)
        #add in the noise image wehre the input scans are
        #replace the data with the overwrited noisified current octomap image

        print(latents.shape)
        
        #now we just iterate through all the coordinates,eveywhere we have a cordinate we put in the noisy new oocumancy
        for idx, z_val in enumerate(input_coordinate[0]):
            #we are iterating though the tuple using the first coord
            x_val = input_coordinate[1][idx]
            y_val = input_coordinate[2][idx]
            # print(latents.shape)
            # print(noisy_images.shape)
            # print(z_val, x_val, y_val)
            #replace the latent value with the new noisified input value
            latents[0][z_val, x_val, y_val] = noisy_images[z_val, x_val, y_val]
        #also iterate through all the coordinates and input our freespace
        for idx, z_val in enumerate(unnoc_coordinate[0]):
            #we are iterating though the tuple using the first coord
            x_val = unnoc_coordinate[1][idx]
            y_val = unnoc_coordinate[2][idx]
            # print(latents.shape)
            # print(noisy_images.shape)
            # print(z_val, x_val, y_val)
            #replace the latent value with the new noisified input value
            latents[0][z_val, x_val, y_val] = noisy_unoc_images[z_val, x_val, y_val]
        # EXPECT THESE 
        # break
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep=t)
        latent_model_input = latent_model_input.to(torch_device)
        # predict the noise residual
        with torch.no_grad():
            noise_pred = model(latent_model_input, t).sample
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond
        # compute the previous noisy sample x_t -> x_t-1
        # t.to(torch_device)
        # noise_pred.to(torch_device)
        # latents.to(torch_device)
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

        ############################################
        #just for viewing the outputs
        ###############################################
        # inpained_points = pointmap_to_pc(latents[0],
        #                                  voxel_size = 0.1,
        #                                  x_y_bounds = [-2, 2],
        #                                   z_bounds = [-1.4, 0.9])
        # pcd_inpaint = o3d.geometry.PointCloud()

        # # print("inpainted shape: ", inpained_points.shape)
        # pcd_inpaint.points = o3d.utility.Vector3dVector(inpained_points)
        # colors = np.zeros((len(np.asarray(pcd_inpaint.points)), 3))
        # R = pcd_inpaint.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
        # pcd_inpaint.rotate(R, center=(0, 0, 0))
        # # colors[:,0] = 1
        # # colors[:,1] = 0
        # # colors[:,2] = 0
        # # pcd_inpaint.colors = o3d.utility.Vector3dVector(colors)
        # # pcd_inpaint.transform(hm_tx_mat)
        # o3d.visualization.draw_geometries([pcd_inpaint])


    #one more inpainting step
    for idx, z_val in enumerate(input_coordinate[0]):
        #we are iterating though the tuple using the first coord
        x_val = input_coordinate[1][idx]
        y_val = input_coordinate[2][idx]
        # print(latents.shape)
        # print(noisy_images.shape)
        # print(z_val, x_val, y_val)
        #replace the latent value with the new noisified input value
        latents[0][z_val, x_val, y_val] = voxel_grid[z_val, x_val, y_val]
    #also iterate through all the coordinates and input our freespace
    for idx, z_val in enumerate(unnoc_coordinate[0]):
        #we are iterating though the tuple using the first coord
        x_val = unnoc_coordinate[1][idx]
        y_val = unnoc_coordinate[2][idx]
        # print(latents.shape)
        # print(noisy_images.shape)
        # print(z_val, x_val, y_val)
        #replace the latent value with the new noisified input value
        latents[0][z_val, x_val, y_val] = 1 - unnoc_grid[z_val, x_val, y_val]
    return latents


def efficient_inpainting_pointmaps_w_freespace(model, noise_scheduler, width, inpainting_target, inpainting_unocc, torch_device = "cpu", denoising_steps = 30, guidance_scale = 3, sample_batch_size = 1, mcmc_steps = 5, step_size = 0.1):
    #set up initialize noise scheudler and noise to be operated on
    noise_scheduler.set_timesteps(denoising_steps, device = torch_device)
    noise = torch.randn(inpainting_target.shape).to(torch_device)
    #load the inpainting target
    voxel_grid = torch.tensor(inpainting_target, dtype=torch.float).to(torch_device)
    unnoc_grid = torch.tensor(inpainting_unocc, dtype = torch.float).to(torch_device)
    # unnoc_coordinate = np.where(unnoc_grid > 0.9)
    # print("shape: ", input_coordinate.shape)

    #generate noise to be diffused
    generator = torch.manual_seed(0)  # Seed generator to create the inital latent noise

    #set sample conditioning, concating uncondtioned noise so we only need to do one forward pass

    #generate progress bar
    #perform diffusion
    # print(noise_scheduler.timesteps)
    generator = torch.manual_seed(0)  
    latents = torch.randn((sample_batch_size, model.config.in_channels, width, width),generator=generator)
    latents = latents.to(torch_device)
    latents = latents * noise_scheduler.init_noise_sigma
    for t in tqdm(noise_scheduler.timesteps):
        #get the noisy scan points
        #this adds noise to voxel grid which is our conditioning targets
        noisy_images = noise_scheduler.add_noise(voxel_grid, noise, timesteps = torch.tensor([t.item()]))
        #WE ACTUALLY NEED TO TURN THIS TO Zeros so we do 1 -
        #I think this 1- operation might mess it up
        noisy_unoc_images = noise_scheduler.add_noise(1 - unnoc_grid, noise, timesteps = torch.tensor([t.item()]))
        # print(latents.shape)
        #add in the noise image wehre the input scans are
        #replace the data with the overwrited noisified current octomap image

        # print(latents.shape)
        
        #now we just iterate through all the coordinates,eveywhere we have a cordinate we put in the noisy new oocumancy
        latents[0][voxel_grid>0.9] = noisy_images[voxel_grid>0.9]
        #also iterate through all the coordinates and input our freespace

        latents[0][unnoc_grid>0.9] = noisy_unoc_images[unnoc_grid>0.9]

        # EXPECT THESE 
        # break
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        # latent_model_input = torch.cat([latents] * 2)
        # latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep=t)
        # latent_model_input = latent_model_input.to(torch_device)
        # predict the noise residual
        with torch.no_grad():
            noise_pred = model(latents, t).sample

        # perform guidance
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        # print(latents.shape)

        # # Perform MCMC sampling (ULA for simplicity)
        # ss = 0.1  # (,)
        # std = (2 * ss)**0.5
        # for _ in range(mcmc_steps):
        #     # Calculate the gradient of log-probability (score) from the model's output
        #     with torch.no_grad():
        #         new_pred = model(latents, t).sample
        #     noise_MCMC = torch.randn_like(new_pred) * std  # (B, 3, H, W)
        #     latents = latents + new_pred*0.1+ noise_MCMC
        #     # print(latents.shape)
        #     # gradient = noise_scheduler.scale_model_input(new__pred, t)
 
        #     # # Langevin step: gradient ascent with noise
        #     # latents = latents + 0.5 * step_size * gradient + torch.sqrt(step_size) * torch.randn_like(latents)

        ############################################
        #just for viewing the outputs
        ###############################################
        # inpained_points = pointmap_to_pc(latents[0],
        #                                  voxel_size = 0.1,
        #                                  x_y_bounds = [-2, 2],
        #                                   z_bounds = [-1.4, 0.9])
        # pcd_inpaint = o3d.geometry.PointCloud()

        # # print("inpainted shape: ", inpained_points.shape)
        # pcd_inpaint.points = o3d.utility.Vector3dVector(inpained_points)
        # colors = np.zeros((len(np.asarray(pcd_inpaint.points)), 3))
        # R = pcd_inpaint.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
        # pcd_inpaint.rotate(R, center=(0, 0, 0))
        # # colors[:,0] = 1
        # # colors[:,1] = 0
        # # colors[:,2] = 0
        # # pcd_inpaint.colors = o3d.utility.Vector3dVector(colors)
        # # pcd_inpaint.transform(hm_tx_mat)
        # o3d.visualization.draw_geometries([pcd_inpaint])


    #one more inpainting step
    latents[0][voxel_grid>0.9] = voxel_grid[voxel_grid>0.9]

    latents[0][unnoc_grid>0.9] = noisy_unoc_images[unnoc_grid>0.9]

    return latents

def multi_efficient_inpainting_pointmaps_w_freespace(model, noise_scheduler, width, inpainting_target, inpainting_unocc, torch_device = "cpu", denoising_steps = 30, guidance_scale = 3, sample_batch_size = 16):
    #set up initialize noise scheudler and noise to be operated on
    noise_scheduler.set_timesteps(denoising_steps, device = torch_device)
    noise = torch.randn(inpainting_target.shape).to(torch_device)
    #load the inpainting target
    voxel_grid = torch.tensor(inpainting_target, dtype=torch.float).to(torch_device)
    unnoc_grid = torch.tensor(inpainting_unocc, dtype = torch.float).to(torch_device)
    # unnoc_coordinate = np.where(unnoc_grid > 0.9)
    # print("shape: ", input_coordinate.shape)

    #generate noise to be diffused
    generator = torch.manual_seed(0)  # Seed generator to create the inital latent noise

    #set sample conditioning, concating uncondtioned noise so we only need to do one forward pass

    #generate progress bar
    #perform diffusion
    # print(noise_scheduler.timesteps)
    generator = torch.manual_seed(0)  
    latents = torch.randn((sample_batch_size, model.config.in_channels, width, width),generator=generator)
    latents = latents.to(torch_device)
    latents = latents * noise_scheduler.init_noise_sigma
    for t in tqdm(noise_scheduler.timesteps):
        #get the noisy scan points
        #this adds noise to voxel grid which is our conditioning targets
        noisy_images = noise_scheduler.add_noise(voxel_grid, noise, timesteps = torch.tensor([t.item()]))
        noisy_images = noisy_images[None,:,:,:]
        #WE ACTUALLY NEED TO TURN THIS TO Zeros so we do 1 -
        #I think this 1- operation might mess it up
        noisy_unoc_images = noise_scheduler.add_noise(1 - unnoc_grid, noise, timesteps = torch.tensor([t.item()]))
        noisy_unoc_images = noisy_unoc_images[None,:,:,:]
        # print(latents.shape)
        #add in the noise image wehre the input scans are
        #replace the data with the overwrited noisified current octomap image

        # print(latents.shape)
        
        #now we just iterate through all the coordinates,eveywhere we have a cordinate we put in the noisy new oocumancy
        latents[:][voxel_grid>0.9] = noisy_images[voxel_grid>0.9]
        #also iterate through all the coordinates and input our freespace
        latents[:][unnoc_grid>0.9] = noisy_unoc_images[unnoc_grid>0.9]
        # EXPECT THESE 
        # break
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        # latent_model_input = torch.cat([latents] * 2)
        # latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep=t)
        # latent_model_input = latent_model_input.to(torch_device)
        # predict the noise residual
        with torch.no_grad():
            noise_pred = model(latents, t).sample
        # perform guidance
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

        ############################################
        #just for viewing the outputs
        ###############################################
        # inpained_points = pointmap_to_pc(latents[0],
        #                                  voxel_size = 0.1,
        #                                  x_y_bounds = [-2, 2],
        #                                   z_bounds = [-1.4, 0.9])
        # pcd_inpaint = o3d.geometry.PointCloud()

        # # print("inpainted shape: ", inpained_points.shape)
        # pcd_inpaint.points = o3d.utility.Vector3dVector(inpained_points)
        # colors = np.zeros((len(np.asarray(pcd_inpaint.points)), 3))
        # R = pcd_inpaint.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
        # pcd_inpaint.rotate(R, center=(0, 0, 0))
        # # colors[:,0] = 1
        # # colors[:,1] = 0
        # # colors[:,2] = 0
        # # pcd_inpaint.colors = o3d.utility.Vector3dVector(colors)
        # # pcd_inpaint.transform(hm_tx_mat)
        # o3d.visualization.draw_geometries([pcd_inpaint])


    #one more inpainting step
    latents[:][voxel_grid>0.9] = noisy_images[voxel_grid>0.9]
    latents[:][unnoc_grid>0.9] = noisy_unoc_images[unnoc_grid>0.9]
    return latents
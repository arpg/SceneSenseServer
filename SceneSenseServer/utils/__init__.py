"""
SceneSenseServer.utils - Utility functions for point cloud processing and diffusion

This module provides utility functions for working with point clouds, point maps,
coordinate transformations, and diffusion-based scene completion.

Functions:
    homogeneous_transform: Generate a homogeneous transformation matrix
    inverse_homogeneous_transform: Invert a homogeneous transformation matrix
    update_points: Filter points based on axis bounds
    set_rgb: Set colors for point cloud visualization
    points_within_distance: Filter points within a distance threshold
    pc_to_pointmap: Convert a point cloud to a pointmap representation
    pointmap_to_pc: Convert a pointmap back to a point cloud
    efficient_inpainting_pointmaps_w_freespace: Apply diffusion model for inpainting
"""

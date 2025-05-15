# SceneSense Server

A Python package for scene understanding, diffusion-based point cloud completion, and frontier detection for robotic exploration.

## Overview

SceneSense Server provides tools for processing and analyzing 3D point cloud data, with a focus on completing occluded or partially observed scenes using diffusion models. It includes functionality for frontier detection, point cloud processing, and a Flask-based server for remote processing.

## Installation

### Using Conda (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/SceneSenseServer.git
cd SceneSenseServer

# Create and activate conda environment
conda env create -f environment.yaml
conda activate SceneSenseServer

# Install in development mode
pip install -e .
```

### Manual Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/SceneSenseServer.git
cd SceneSenseServer

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Dependencies
The main dependencies include:
- Flask for the server implementation
- Open3D for point cloud processing
- PyTorch and Diffusers for the diffusion model
- NumPy, SciPy for numerical operations
- CUDA support is recommended for faster model inference

All dependencies will be automatically installed when setting up the conda environment or using pip.

## Package Structure

The package is organized into several modules:

- `SceneSenseServer.core`: Core functionality including frontier detection and scene analysis
  - `frontier_finder.py`: Implements algorithms for identifying and clustering frontier points
  
- `SceneSenseServer.server`: Flask server implementations for various services
  - `diffusion_server.py`: Main server implementation for diffusion-based point cloud completion
  - `client_test.py`: Client for testing server functionality
  
- `SceneSenseServer.utils`: Utility functions and helper tools
  - Includes functions for point cloud manipulation, transformation, and visualization

## Usage

### Running the Diffusion Server
```bash
# Activate the conda environment if using conda
conda activate SceneSenseServer

# Run the server
python -m SceneSenseServer.server.diffusion_server
```

The server will start on http://0.0.0.0:5000 by default.

### Testing the Server
You can test the server with the provided client:

```bash
# Make sure the server is running first
python -m SceneSenseServer.server.client_test
```

### Using the Frontier Finder
```python
from SceneSenseServer.core import frontier_finder
import open3d as o3d

# Load point cloud data
occupied_pcd = o3d.io.read_point_cloud("path/to/occupied.pcd")
unoccupied_pcd = o3d.io.read_point_cloud("path/to/unoccupied.pcd")

# Initialize frontier finder
finder = frontier_finder.FrontierFinder()

# Find frontier points
frontier_pcd, frontier_centroids = finder.find_frontiers(occupied_pcd, unoccupied_pcd)

# Visualize results
o3d.visualization.draw_geometries([frontier_pcd])
```

### Using the Diffusion Model Programmatically
```python
import torch
from diffusers import DDPMScheduler, UNet2DModel
from SceneSenseServer.utils import utils
import numpy as np

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet2DModel.from_pretrained("alre5639/frontier_diff_no_cond_try_2").to(device)
scheduler = DDPMScheduler(num_train_timesteps=1000)

# Prepare pointmap data
# ... (load and prepare pointmaps)

# Apply diffusion model
inpainted_pm = utils.efficient_inpainting_pointmaps_w_freespace(
    model, 
    scheduler, 
    64, 
    occupied_pointmap, 
    unoccupied_pointmap,
    torch_device=device,
    denoising_steps=30
)

# Convert to point cloud
inpainted_points = utils.pointmap_to_pc(
    inpainted_pm[0], 
    voxel_size=0.1, 
    x_y_bounds=[-3.2, 3.2], 
    z_bounds=[-1.3, 1.0]
)
```

## API Documentation

### Core Module

#### FrontierFinder
The `FrontierFinder` class provides methods for identifying exploration frontiers in point cloud data.
- `load_point_clouds()`: Loads and preprocesses point cloud data
- `transform_point_clouds()`: Transforms point clouds using odometry data
- `find_frontiers()`: Identifies and clusters frontier points

### Server Module

#### Diffusion Server
The diffusion server provides two main endpoints:
- `/`: Root endpoint to check if the server is running
- `/diffusion`: POST endpoint for receiving point cloud data and returning completed scenes

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

SceneSense Server was developed by Alec Reed, Lorin Achey, and Brendan Crowe.

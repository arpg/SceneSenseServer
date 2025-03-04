# SceneSense

A Python package for scene understanding and diffusion-based completion.

## Installation

### From Source
```bash
# Clone the repository
git clone https://github.com/yourusername/SceneSenseServer.git
cd SceneSenseServer

# Install in development mode
pip install -e .
```

### Dependencies
All dependencies will be automatically installed when installing the package. If you want to install them manually, you can use:
```bash
pip install -r requirements.txt
```

## Package Structure

The package is organized into several modules:

- `SceneSenseServer.core`: Core functionality including frontier detection and scene analysis
- `SceneSenseServer.server`: Flask server implementations for various services
- `SceneSenseServer.utils`: Utility functions and helper tools
- `SceneSenseServer.training`: Training scripts and data generation utilities

## Usage

### Starting the Server
```python
from SceneSenseServer.server import flask_diffusion_server

# Start the diffusion server
flask_diffusion_server.run()
```

### Running Scene Analysis
```python
from SceneSenseServer.core import frontier_finder

# Initialize and run frontier detection
finder = frontier_finder.FrontierFinder()
frontiers = finder.find_frontiers(point_cloud)
```

## License

[Your chosen license]

## Contributing

[Your contribution guidelines]

from SceneSenseServer.core.frontier_finder import FrontierFinder
from SceneSenseServer.server import flask_diffusion_server
from SceneSenseServer.utils import utils

print("Successfully imported SceneSenseServer modules!")

# Create a FrontierFinder instance with custom data paths
finder = FrontierFinder(
    data_dir="data/range_max",  # Update this path to your data directory
    odom_dir="/hdd/spot_diff_data/odometry/odometry",  # Update this path to your odometry directory
)

# The actual data loading and processing can be done when needed:
# occ_pcd, unocc_pcd = finder.load_point_clouds()
# occ_pcd, unocc_pcd = finder.transform_point_clouds(occ_pcd, unocc_pcd)
# frontier_pcd, median_points = finder.find_frontiers(occ_pcd, unocc_pcd)

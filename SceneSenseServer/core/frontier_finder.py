import os

import numpy as np
import open3d as o3d
from natsort import natsorted
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN

from SceneSenseServer.utils import utils


def is_surrounded(point, close_points):
    """Check if a point is surrounded by other points."""
    # Calculate differences between the point and its neighbors
    differences = close_points - point

    # Check for points in positive and negative directions for each axis
    positive_x = np.any(differences[:, 0] > 0.1)
    negative_x = np.any(differences[:, 0] < -0.1)
    positive_y = np.any(differences[:, 1] > 0.1)
    negative_y = np.any(differences[:, 1] < -0.1)
    positive_z = np.any(differences[:, 2] > 0.1)
    negative_z = np.any(differences[:, 2] < -0.1)

    # Return True if there are positive and negative values for x, y, and z
    return (
        positive_x
        and negative_x
        and positive_y
        and negative_y
        and positive_z
        and negative_z
    )


def categorize_points(points, categories):
    """Group points by their categories."""
    categories_dict = {}
    for point, category in zip(points, categories):
        if category not in categories_dict:
            categories_dict[category] = []
        categories_dict[category].append(point)
    return categories_dict


def calculate_median_points(categories_dict):
    """Calculate median points for each category."""
    medians = []
    for category in categories_dict:
        points = np.array(categories_dict[category])
        median = np.median(points, axis=0)
        medians.append(median)
    return np.array(medians)


class FrontierFinder:
    def __init__(self, data_dir=None, odom_dir=None):
        self.data_dir = data_dir or "data/range_max"
        self.odom_dir = odom_dir or "/hdd/spot_diff_data/odometry/odometry"

    def load_point_clouds(self, occ_file="running_occ.pcd", unocc_file="test_unoc.pcd"):
        """Load and preprocess point clouds."""
        occ_pcd_path = os.path.join(self.data_dir, occ_file)
        unocc_pcd_path = os.path.join(self.data_dir, unocc_file)

        occ_pcd = o3d.io.read_point_cloud(occ_pcd_path)
        occ_pcd = utils.update_points(occ_pcd, -1.3, 2, 2)

        unocc_pcd = o3d.io.read_point_cloud(unocc_pcd_path)
        unocc_pcd = utils.update_points(unocc_pcd, -1.3, 2, 2)

        return occ_pcd, unocc_pcd

    def transform_point_clouds(self, occ_pcd, unocc_pcd, odom_idx=409):
        """Transform point clouds using odometry data."""
        odom_file_names = natsorted(os.listdir(self.odom_dir))
        pose = np.load(os.path.join(self.odom_dir, odom_file_names[odom_idx]))
        rotation_obj = Rotation.from_rotvec(pose[3::])
        hm_tx_mat = utils.homogeneous_transform(pose[0:3], rotation_obj.as_quat())

        occ_pcd.transform(utils.inverse_homogeneous_transform(hm_tx_mat))
        unocc_pcd.transform(utils.inverse_homogeneous_transform(hm_tx_mat))

        occ_pcd = utils.update_points(occ_pcd, 0, 10, 1)
        unocc_pcd = utils.update_points(unocc_pcd, 0, 10, 1)

        return occ_pcd, unocc_pcd

    def find_frontiers(self, occ_pcd, unocc_pcd, eps=0.3, min_samples=5):
        """Find frontier points in the point clouds."""
        occ_pcd = utils.set_rgb(occ_pcd)
        unocc_pcd = utils.set_rgb(unocc_pcd, 0)

        # Convert to numpy arrays
        occ_points = np.asarray(occ_pcd.points)
        unocc_points = np.asarray(unocc_pcd.points)
        all_points = np.append(occ_points, unocc_points, axis=0)

        # Build KD-tree and find nearest neighbors
        kdtree = KDTree(all_points)
        dist, points = kdtree.query(
            unocc_points, 7
        )  # need to query 7 because it includes itself

        # Find frontier points
        front_point_arr = np.empty((0, 3), float)
        for unoc_point, near_point_idx in zip(unocc_points, points):
            front_point = is_surrounded(unoc_point, all_points[near_point_idx])
            if front_point == False:
                unoc_point = unoc_point[None, :]
                front_point_arr = np.append(front_point_arr, unoc_point, axis=0)

        # Cluster frontier points
        model = DBSCAN(eps=eps, min_samples=min_samples).fit(front_point_arr)

        # Create colored point cloud for visualization
        cluster_colors = np.zeros((len(front_point_arr), 3))
        for idx, cluster in enumerate(model.labels_):
            np.random.seed(cluster + 1)
            cluster_colors[idx] = np.random.rand(3)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(front_point_arr)
        pcd.colors = o3d.utility.Vector3dVector(cluster_colors)

        # Compute cluster centroids
        category_dict = categorize_points(front_point_arr, model.labels_ + 1)
        median_fronts = calculate_median_points(category_dict)

        return pcd, median_fronts

import numpy as np
import open3d as o3d
from natsort import natsorted
import utils.utils as utils
from scipy.spatial.transform import Rotation
import os
import copy

# full_pcd_path = "/hdd/spot_diff_data/IRL.pcd"
full_pcd_path = "data/full_irl/running_occ.pcd"
# full_pcd_path = "data/CSEL/running_occ.pcd"
# full_pcd_path = "/hdd/spot_diff_data/IRL.pcd"
# full_pcd_path = "test_unoc.pcd"
full_pcd = o3d.io.read_point_cloud(full_pcd_path)
# odom_path = "/hdd/spot_diff_data/CSEL/CSEL/odometry"
odom_path = "/hdd/spot_diff_data/odometry/odometry/"

odom_file_names = natsorted(os.listdir(odom_path))

for idx, odom_file in enumerate(odom_file_names):

    pose = np.load(os.path.join(odom_path,odom_file))
    # coor = o3d.geometry.TriangleMesh.create_coordinate_frame()

    rotation_obj = Rotation.from_rotvec(pose[3::])
    hm_tx_mat = utils.homogeneous_transform(pose[0:3], rotation_obj.as_quat())

    tx_pcd = copy.deepcopy(full_pcd)
    tx_pcd.transform(utils.inverse_homogeneous_transform(hm_tx_mat))


    local_points = utils.points_within_distance(0.0,0.0,np.asarray(tx_pcd.points),3.2)
    local_pcd = o3d.geometry.PointCloud()
    local_pcd.points = o3d.utility.Vector3dVector(local_points)
    local_pcd = utils.update_points(local_pcd,-1.3,1,2)

    local_octomap_pm = utils.pc_to_pointmap(np.asarray(local_pcd.points), 
                                            voxel_size = 0.1,
                                            x_y_bounds = [-3.2, 3.2],
                                            z_bounds = [-1.3, 1.0])
    print(local_octomap_pm.shape)
    np.save("/hdd/spot_diff_data/training_data/IRL/"+str(idx)+".npy", local_octomap_pm)
    # o3d.visualization.draw_geometries([local_pcd])
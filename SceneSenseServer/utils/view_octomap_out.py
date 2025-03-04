import open3d as o3d
from natsort import natsorted
import os
import numpy as np
import utils.utils as utils
from scipy.spatial.transform import Rotation

points_path = "data/range_max/running_occ.pcd"
# point_file_names = natsorted(os.listdir(points_path))
pcd = o3d.io.read_point_cloud(points_path)
#get the points
pcd = utils.update_points(pcd,-1.3,1,2)
# pcd = utils.update_points(pcd,0,10,1)

coor = o3d.geometry.TriangleMesh.create_coordinate_frame()

#load the current pose
odom_path = "/hdd/spot_diff_data/odometry/odometry"
odom_file_names = natsorted(os.listdir(odom_path))

pose = np.load(os.path.join(odom_path,odom_file_names[480]))
rotation_obj = Rotation.from_rotvec(pose[3::])
hm_tx_mat = utils.homogeneous_transform(pose[0:3], rotation_obj.as_quat())
coor.transform(hm_tx_mat)
o3d.visualization.draw_geometries([pcd, coor])
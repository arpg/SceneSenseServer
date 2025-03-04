import open3d as o3d
from natsort import natsorted
import os
import numpy as np
import utils.utils as utils
from scipy.spatial.transform import Rotation

points_path = "data/range_max/running_occ.pcd"
unoc_points = "data/range_max/test_unoc.pcd"
# point_file_names = natsorted(os.listdir(points_path))
pcd = o3d.io.read_point_cloud(points_path)
pcd = utils.update_points(pcd,-1.3,1,2)
unoc_pcd = o3d.io.read_point_cloud(unoc_points)
unoc_pcd = utils.update_points(unoc_pcd,-1.3,1,2)

# pcd = utils.update_points(pcd,0,10,1)

coor = o3d.geometry.TriangleMesh.create_coordinate_frame()

#load the current pose
odom_path = "/hdd/spot_diff_data/odometry/odometry"
odom_file_names = natsorted(os.listdir(odom_path))

pose = np.load(os.path.join(odom_path,odom_file_names[480]))
rotation_obj = Rotation.from_rotvec(pose[3::])
hm_tx_mat = utils.homogeneous_transform(pose[0:3], rotation_obj.as_quat())
#translation to frontier
frontier_trans = utils.homogeneous_transform(np.array([1.0, 0.0, 0]), np.array([0,0,0,1]))

pcd.transform(utils.inverse_homogeneous_transform(hm_tx_mat))
unoc_pcd.transform(utils.inverse_homogeneous_transform(hm_tx_mat))
#clean up the spurious detections for example
pcd = utils.update_points(pcd,-100,2,1)
unoc_pcd = utils.update_points(unoc_pcd,-100,2,1)
pcd.transform(utils.inverse_homogeneous_transform(frontier_trans))
unoc_pcd.transform(utils.inverse_homogeneous_transform(frontier_trans))
# o3d.visualization.draw_geometries([pcd, coor])

#get the local info. 
local_points = utils.points_within_distance(0.0,0.0,np.asarray(pcd.points),2.0)
local_points = local_points[local_points[:,2] < 1.0]
local_points_unoc = utils.points_within_distance(0.0,0.0,np.asarray(unoc_pcd.points),2.0)
local_points_unoc = local_points_unoc[local_points_unoc[:,2] < 1.0]
local_pcd = o3d.geometry.PointCloud()
local_pcd.points = o3d.utility.Vector3dVector(local_points)
local_unoc_pcd = o3d.geometry.PointCloud()
local_unoc_pcd.points = o3d.utility.Vector3dVector(local_points_unoc)
local_pcd = utils.set_rgb(local_pcd)
coor = o3d.geometry.TriangleMesh.create_coordinate_frame()
diff_points = o3d.io.read_point_cloud("test_inpainted_pcd.pcd")
diff_points = utils.set_rgb(diff_points, 0)
o3d.visualization.draw_geometries([local_pcd, diff_points, coor])
import numpy as np
import open3d as o3d
from natsort import natsorted
import utils.utils as utils
from scipy.spatial.transform import Rotation
import os
import copy
import utils.utils as utils
def remove_close_points(pcd, diff_pcd, threshold=0.1):
    # Convert Open3D point clouds to numpy arrays
    pcd_points = np.asarray(pcd.points)
    diff_pcd_points = np.asarray(diff_pcd.points)

    # Create a KDTree for the pcd points
    pcd_kdtree = o3d.geometry.KDTreeFlann(pcd)

    # List to hold indices of points to remove
    indices_to_remove = []

    # Iterate through each point in diff_pcd
    for i, point in enumerate(diff_pcd_points):
        [k, idx, _] = pcd_kdtree.search_radius_vector_3d(point, threshold)
        if k > 0:
            indices_to_remove.append(i)

    # Remove points from diff_pcd
    diff_pcd_points_filtered = np.delete(diff_pcd_points, indices_to_remove, axis=0)
    return diff_pcd_points_filtered
    # # Update diff_pcd with the filtered points
    # diff_pcd.points = o3d.utility.Vector3dVector(diff_pcd_points_filtered)

    # return diff_pcd
def remove_points_within_distance(points, distance):
    """
    Remove points within a certain distance from the origin (0,0,0).

    Parameters:
    points (np.ndarray): An nx3 array of 3D points.
    distance (float): The distance threshold.

    Returns:
    np.ndarray: The filtered array of 3D points.
    """
    # Compute the squared distance from the origin for each point
    squared_distances = np.sum(points**2, axis=1)
    
    # Compute the squared distance threshold to avoid computing square root
    squared_distance_threshold = distance**2
    
    # Filter points based on the squared distance
    filtered_points = points[squared_distances > squared_distance_threshold]
    
    return filtered_points

coor = o3d.geometry.TriangleMesh.create_coordinate_frame()
coor.scale(0.5, center=coor.get_center())

#transform the map
frontier_trans = utils.homogeneous_transform(np.array([9.5, 1.0, 0.3]), np.array([0,0,0,1]))
# #load the spot
# spot = o3d.io.read_triangle_mesh("/hdd/ce_kitti_features/spot.ply")
# # Paint each vertex yellow (R=1, G=1, B=0)
# yellow_color = np.array([1, 1, 0])
# spot.vertex_colors = o3d.utility.Vector3dVector(np.tile(yellow_color, (len(spot.vertices), 1)))

points_path = "/hdd/IJRR_data/corner_maps/baseline_pcds"
point_file_names = natsorted(os.listdir(points_path))
pcd = o3d.io.read_point_cloud(os.path.join(points_path,point_file_names[1]))
#transform pcd
pcd.transform(utils.inverse_homogeneous_transform(frontier_trans))
#remove celing
points = np.asarray(pcd.points)
points = points[points[:,2] < 1.0]
points = points[points[:,2] > -1.2]
#filter out any points within some distance of 0,0,0
points = remove_points_within_distance(points,0.6)
pcd.points = o3d.utility.Vector3dVector(points)

#color points
pcd_colors = np.zeros((len(np.asarray(pcd.points)), 3))
max_z = np.max(np.asarray(pcd.points)[:,2], axis = 0)
min_z = np.min(np.asarray(pcd.points)[:,2], axis = 0)

#create scaler constant
# colors[:,0] = ((np.asarray(local_occ_pcd.points)[:,1] + 1.45)/(max_z - min_z))
pcd_colors[:,1] = ((np.asarray(pcd.points)[:,2] - min_z)/(max_z - min_z))*(1 - 0.3) + 0.3
pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
pcd_vox = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.15)

###################################3
#load the diffused pcd:
######################################
diff_points_path = "/hdd/IJRR_data/corner_maps/diffused_pcds"
diff_point_file_names = natsorted(os.listdir(diff_points_path))
pcd_diff = o3d.io.read_point_cloud(os.path.join(diff_points_path,diff_point_file_names[-10]))
pcd_diff.transform(utils.inverse_homogeneous_transform(frontier_trans))
#remove celing
points = np.asarray(pcd_diff.points)
points = points[points[:,2] < 1.0]
points = points[points[:,2] > -1.2]
points = utils.points_within_distance(0,0,points,3.3)
points = remove_points_within_distance(points,0.6)
pcd_diff.points = o3d.utility.Vector3dVector(points)
diff_point_filtered = remove_close_points(pcd, pcd_diff)
#romve points some distance from the spot
# diff_points_filtered = utils.points_within_distance(0,0,diff_point_filtered,3.3)

# diff_point_filtered[:,0] = diff_point_filtered[:,0] + 0.1
diff_pcd_filtered = o3d.geometry.PointCloud()
diff_pcd_filtered.points = o3d.utility.Vector3dVector(diff_point_filtered)
# #color points
colors = np.zeros((len(np.asarray(diff_pcd_filtered.points)), 3))
max_z = np.max(np.asarray(diff_pcd_filtered.points)[:,2], axis = 0)
min_z = np.min(np.asarray(diff_pcd_filtered.points)[:,2], axis = 0)

# #create scaler constant
# colors[:,0] = ((np.asarray(local_occ_pcd.points)[:,1] + 1.45)/(max_z - min_z))
colors[:,0] = ((np.asarray(diff_pcd_filtered.points)[:,2] - min_z)/(max_z - min_z))*(1 - 0.3) + 0.3
diff_pcd_filtered.colors = o3d.utility.Vector3dVector(colors)

pcd_diff_vox = o3d.geometry.VoxelGrid.create_from_point_cloud(diff_pcd_filtered, voxel_size=0.15)
# o3d.visualization.draw_geometries([pcd_vox, coor])
# o3d.visualization.draw_geometries([pcd_vox,pcd_diff_vox, coor])

pcd_points = np.asarray(copy.deepcopy(pcd).points)
print(len(pcd_points))
print(len(pcd_colors))
diff_points = np.asarray(diff_pcd_filtered.points)
pcd_points = np.append(pcd_points,diff_points, axis = 0)
final_colors = np.append(pcd_colors,colors, axis = 0)
final_pcd = o3d.geometry.PointCloud()
final_pcd.points = o3d.utility.Vector3dVector(pcd_points)
final_pcd.colors = o3d.utility.Vector3dVector(final_colors)
comb_vox = o3d.geometry.VoxelGrid.create_from_point_cloud(final_pcd, voxel_size=0.15)
view_params = None


def key_callback(vis):
    global step
    try: step
    except NameError: step = 10

    print("step forward step " + str(step))
    # pcd = o3d.io.read_point_cloud(os.path.join(points_path,point_file_names[step]))
    ctr  = vis.get_view_control()
    view_param =ctr.convert_to_pinhole_camera_parameters()
    vis.clear_geometries()
    vis.add_geometry(pcd_vox)
    vis.add_geometry(coor)
    ctr.convert_from_pinhole_camera_parameters(view_param)
    step += 1
def key_callback_backwards(vis):
    global step
    try: step
    except NameError: step = 10
    
    print("step_back step " + str(step))
    # pcd = o3d.io.read_point_cloud(os.path.join(points_path,point_file_names[step]))
    ctr  = vis.get_view_control()
    view_param =ctr.convert_to_pinhole_camera_parameters()
    vis.clear_geometries()
    vis.add_geometry(pcd_diff_vox)
    vis.add_geometry(pcd_vox)
    vis.add_geometry(coor)
    ctr.convert_from_pinhole_camera_parameters(view_param)
    step -= 1


vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()
vis.add_geometry(pcd)
# vis.add_geometry(spot)
vis.register_key_callback(68, key_callback) #65 is a 39 shoudl be forward arrow
vis.register_key_callback(65, key_callback_backwards) #66 is b 37 shoudl be backwards arrow
vis.run()
vis.destroy_window()
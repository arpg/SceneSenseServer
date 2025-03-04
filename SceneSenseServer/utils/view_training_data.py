import numpy as np
import open3d as o3d
from natsort import natsorted
import utils.utils as utils
from scipy.spatial.transform import Rotation
import os
import copy

# points_path = "/hdd/spot_diff_data/gt_point_maps/point_clouds"
# point_file_names = natsorted(os.listdir(points_path))
# pcd = o3d.io.read_point_cloud(os.path.join(points_path,point_file_names[10]))

#load the pm
pcd = o3d.io.read_point_cloud("/hdd/spot_diff_data/updated_training_data/7th_floor/" + str(1) + ".pcd")
def key_callback(vis):
    global step
    try: step
    except NameError: step = 0

    print("step forward step " + str(step))
    pcd = o3d.io.read_point_cloud("/hdd/spot_diff_data/updated_training_data/7th_floor/" + str(step) + ".pcd")
    ctr  = vis.get_view_control()
    view_param =ctr.convert_to_pinhole_camera_parameters()
    vis.clear_geometries()
    vis.add_geometry(pcd)
    # vis.add_geometry(coor)
    ctr.convert_from_pinhole_camera_parameters(view_param)
    step += 1
def key_callback_backwards(vis):
    global step
    try: step
    except NameError: step = 10
    
    print("step_back step " + str(step))
    pcd = o3d.io.read_point_cloud("/hdd/spot_diff_data/updated_training_data/7th_floor/" + str(step) + ".pcd")
    ctr  = vis.get_view_control()
    view_param =ctr.convert_to_pinhole_camera_parameters()
    vis.clear_geometries()
    vis.add_geometry(pcd)
    # vis.add_geometry(coor)
    ctr.convert_from_pinhole_camera_parameters(view_param)
    step -= 1


vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()
vis.add_geometry(pcd)
vis.register_key_callback(68, key_callback) #65 is a 39 shoudl be forward arrow
vis.register_key_callback(65, key_callback_backwards) #66 is b 37 shoudl be backwards arrow
vis.run()
vis.destroy_window()
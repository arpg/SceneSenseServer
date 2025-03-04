import open3d as o3d
from natsort import natsorted
import os
import numpy as np

points_path = "/hdd/spot_diff_data/point_clouds/point_clouds"
point_file_names = natsorted(os.listdir(points_path))
pcd = o3d.io.read_point_cloud(os.path.join(points_path,point_file_names[10]))
coor = o3d.geometry.TriangleMesh.create_coordinate_frame()
# colors = np.zeros((len(np.asarray(pcd.points)), 3))
# colors[:,0] = 1
# pcd.colors = o3d.utility.Vector3dVector(colors)
# print(np.asarray(pcd.points).shape)
# o3d.visualization.draw_geometries([pcd])
# print(first_map.shape)
def key_callback(vis):
    global step
    try: step
    except NameError: step = 10

    print("step forward step " + str(step))
    pcd = o3d.io.read_point_cloud(os.path.join(points_path,point_file_names[step]))
    ctr  = vis.get_view_control()
    view_param =ctr.convert_to_pinhole_camera_parameters()
    vis.clear_geometries()
    vis.add_geometry(pcd)
    vis.add_geometry(coor)
    ctr.convert_from_pinhole_camera_parameters(view_param)
    step += 1
def key_callback_backwards(vis):
    global step
    try: step
    except NameError: step = 10
    
    print("step_back step " + str(step))
    pcd = o3d.io.read_point_cloud(os.path.join(points_path,point_file_names[step]))
    ctr  = vis.get_view_control()
    view_param =ctr.convert_to_pinhole_camera_parameters()
    vis.clear_geometries()
    vis.add_geometry(pcd)
    vis.add_geometry(coor)
    ctr.convert_from_pinhole_camera_parameters(view_param)
    step -= 1


vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()
vis.add_geometry(pcd)
vis.register_key_callback(68, key_callback) #65 is a 39 shoudl be forward arrow
vis.register_key_callback(65, key_callback_backwards) #66 is b 37 shoudl be backwards arrow
vis.run()
vis.destroy_window()
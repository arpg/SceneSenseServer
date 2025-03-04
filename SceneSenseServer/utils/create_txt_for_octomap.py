import open3d as o3d
from natsort import natsorted
import os
import numpy as np
import copy

points_path = "/hdd/spot_diff_data/CSEL/CSEL/point_clouds"
point_file_names = natsorted(os.listdir(points_path))

odom_path = "/hdd/spot_diff_data/CSEL/CSEL/odometry"
odom_file_names = natsorted(os.listdir(odom_path))

pose = np.load(os.path.join(odom_path,odom_file_names[0]))
points = o3d.io.read_point_cloud(os.path.join(points_path,point_file_names[10]))

#need to remove names that dont appear in both
odom_pruned_file_names = []
point_pruned_file_names = []

for odom_name in odom_file_names:
    name_flag = False
    for point_name in point_file_names:
        if odom_name[0:-3] == point_name[0:-3]:
            odom_pruned_file_names.append(odom_name)
            point_pruned_file_names.append(point_name)
            

print(len(odom_pruned_file_names), " ", len(point_pruned_file_names))

#now create a giant txt file for octomap

for idx, (point_name, odom_name) in enumerate(zip(point_pruned_file_names,odom_pruned_file_names)):
    #load the data
    if idx > 100:
        print(idx, point_name,odom_name)
        pose = np.load(os.path.join(odom_path,odom_name))
        points = o3d.io.read_point_cloud(os.path.join(points_path,point_name))
        points_np = copy.deepcopy(np.asarray(points.points))

        #write to the textfile
        #first need to write the opening string
        f = open("data/CSEL/sample_octomap_running.txt", "a")
        write_str = "NODE " + str(pose[0]) +" " + str(pose[1]) + " " + str(pose[2]) + " " + str(pose[3]) + " " + str(pose[4])  + " " + str(pose[5])  + "\n"
        # print(write_str)
        f.write(write_str)
        f.close()

        #now need to add all the points
        with open("data/CSEL/sample_octomap_running.txt", "ab") as f:
            np.savetxt(f, points_np)
        f.close()
        #now need to break at 390
        # if idx == 480:
        #     break

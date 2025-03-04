import subprocess

#########################3
# build the octomap
########################3
# Replace 'your_program' with the actual name of your compiled C++ program
cpp_program = '/home/arpg/Documents/octomap/bin/log2graph'
# Command-line arguments for your C++ program
cpp_arguments = ["data/CSEL/sample_octomap_running.txt", "data/CSEL/running_octograph.graph"]
# Use subprocess to run the C++ program with arguments
subprocess.run([cpp_program] + cpp_arguments)
print("\ngraph made\n")
# build the tree
# sleep(2)
cpp_program = '/home/arpg/Documents/octomap/bin/graph2tree'
# Command-line arguments for your C++ program
cpp_arguments = ["-i", "data/CSEL/running_octograph.graph", "-o", "data/CSEL/octree.bt", "-m", "15", "-clamping", "0.1", "0.9", "-sensor", "0.4", "0.85"] #these are the subt values, honestly look super bad something is werid
# Use subprocess to run the C++ program with arguments
subprocess.run([cpp_program] + cpp_arguments)
print("\tree made\n")
#export occupied and unocupied frames
cpp_program = '/home/arpg/Documents/octomap/bin/octree2pointcloud'
# Command-line arguments for your C++ program
cpp_arguments = ["data/CSEL/octree.bt", "data/CSEL/running_occ.pcd"]
# Use subprocess to run the C++ program with arguments
subprocess.run([cpp_program] + cpp_arguments)

import numpy as np
import os
from natsort import natsorted
import copy
import cv2
dir_path = "/home/cc/updated_gt_maps"
gt_files = natsorted(os.listdir(dir_path))
# Calculate padding
pad_height = (64 - 44) // 2  # Padding for height
pad_width = (64 - 44) // 2   # Padding for width
for gt_file in gt_files:
    data = np.load(os.path.join(dir_path,gt_file))
    # print(data.shape)
    padded_array = np.pad(data, pad_width=((0, 0), (pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    # print(padded_array.shape)
    # for i, img in enumerate(np.asarray(padded_array)):
    #         #normalize the outputs to 255 in each pixel
    #         output = copy.deepcopy(img) * 255
    #         #dupicate it to be an image
    #         output = np.repeat(output[:, :, np.newaxis], 3, axis=2)
            
    #         #save it as a cv2 image
    #         cv2.imwrite("data/" + str(i) + ".png", output )
    
    # rewrite the file
    # np.save(os.path.join(dir_path,gt_file), padded_array)
    # break
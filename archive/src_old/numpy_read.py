# read numpy files
import numpy as np
import os

numpy_dirs = r"E:\dataset\tvb-hksl-news\keypoints"

# recursively read all numpy files in the directory and concatenate them
def read_numpy_files(dir):
    keypoints = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".npy"):
                keypoints.append(np.load(os.path.join(root, file)))
    return np.concatenate(keypoints)

keypoints = read_numpy_files(numpy_dirs)
print(np.max(keypoints))
print(np.min(keypoints))
# count the number of unique values in the array
print(len(np.unique(keypoints)))
# A debug script to read a single keypoint file and output it to a text file.

import os
import numpy as np
import json

def read_keypoint_file(file: str):
    keypoints = np.load(file)
    return keypoints

def write_keypoint_file(file: str, keypoints: np.ndarray):
    with open(file, 'w') as f:
        for i, slice_2d in enumerate(keypoints):
            f.write(f"Slice {i}:\n")
            np.savetxt(f, slice_2d, fmt='%.6f')
            f.write("\n")

def main():
    keypoint_file = r"E:\dataset\tvb-hksl-news\keypoints\2020-01-16\000453-000550.npy"
    keypoints = read_keypoint_file(keypoint_file)
    write_keypoint_file("keypoints.txt", keypoints)

if __name__ == "__main__":
    main()
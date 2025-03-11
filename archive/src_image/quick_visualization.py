# given a keypoint file, visualize the keypoints on a black background
# keypoints are normalized to the range [0, 1]

import cv2
import numpy as np
import os

def visualize_keypoint(label, id):
    canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
    keypoints = np.load(os.path.join(r"F:\dataset\minified_2\keypoints", label, f"{id}.npy"))
    for keypoint in keypoints:
        x, y, _ = keypoint
        x = int(x * 1920)
        y = int(y * 1080)
        cv2.circle(canvas, (x, y), 5, (0, 255, 0), -1)
    cv2.imshow(f"{label}_{id}", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

for label in os.listdir(r"F:\dataset\minified_2\keypoints"): 
    for id in range(30):
        visualize_keypoint(label, id)
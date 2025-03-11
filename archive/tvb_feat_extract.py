KEYPOINT_INPUT_DIR = r"F:\dataset\tvb-hksl-news\keypoints_mediapipe"
KEYPOINT_OUTPUT_DIR = r"F:\dataset\tvb-hksl-news\keypoints_mediapipe_feat_select"

import os

import numpy as np
from mp_util_legacy import STATIC_KEYPOINTS_INDEX

if __name__ == "__main__":
    os.makedirs(KEYPOINT_OUTPUT_DIR, exist_ok=True)
    for date in os.listdir(KEYPOINT_INPUT_DIR):
        os.makedirs(os.path.join(KEYPOINT_OUTPUT_DIR, date), exist_ok=True)
        for name in os.listdir(os.path.join(KEYPOINT_INPUT_DIR, date)):
            keypoints = np.load(os.path.join(KEYPOINT_INPUT_DIR, date, name))
            keypoints = keypoints[:, STATIC_KEYPOINTS_INDEX]
            np.save(os.path.join(KEYPOINT_OUTPUT_DIR, date, name), keypoints)
            print(f"Processed: {date}/{name}")
import cv2
import numpy as np
import os

from config import config_parser
from mp_util_legacy import mediapipe_extract_multiple

def main(config_path: str):
    config = config_parser(config_path)

    for label in os.listdir(config.paths.dataset):
        os.makedirs(os.path.join(config.paths.keypoints, label), exist_ok=True)
        for i in range(config.sequence.count):
            video_path = os.path.join(config.paths.dataset, label, f"{i}.avi")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Cannot open video {video_path}")
                continue
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            keypoints = mediapipe_extract_multiple(frames, visualize=False)
            keypoint_path = os.path.join(config.paths.keypoints, label, f"{i}.npy")
            np.save(keypoint_path, keypoints)
            print(f"Saved {keypoint_path}")
            cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main("config/config_clip.yaml" if os.name == "nt" else "config/config_clip_linux.yaml")
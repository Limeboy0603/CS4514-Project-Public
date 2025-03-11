import cv2
import numpy as np
import os

from config import config_parser
from archive.cv_util import preprocess_image
from mp_util import init_landmarkers, mediapipe_detect_single, mediapipe_extract_single, draw_landmarks

if __name__ == "__main__":
    config = config_parser("config/config_image.yaml")

    labels = os.listdir(config.paths.dataset)

    for label in labels:
        image_save_dir = os.path.join(config.paths.dataset, label)
        os.makedirs(os.path.join(config.paths.keypoints, label), exist_ok=True)
        os.makedirs(os.path.join(config.paths.keypoints, "..", "visualization", label), exist_ok=True)
        for idx, image in enumerate(os.listdir(image_save_dir)):
            frame = cv2.imread(os.path.join(image_save_dir, image))

            # without transformation
            # landmarkers = init_landmarkers()
            # # mp_results = mediapipe_detect_single(frame, landmarkers, 0)
            # for i in range(10):
            #     # move the frame to the right by 1 pixel if i%2 == 0
            #     if i % 2 == 0:
            #         frame = np.roll(frame, 1, axis=1)
            #     # move it back to the left by 1 pixel if i%2 == 1
            #     else:
            #         frame = np.roll(frame, -1, axis=1)
            #     mp_results = mediapipe_detect_single(frame, landmarkers, i+1)
            # del landmarkers
            # keypoints = mediapipe_extract_single(mp_results)
            # np.save(os.path.join(config.paths.keypoints, label, f"{idx}.npy"), keypoints)

            # frame_copy = frame.copy()
            # draw_landmarks(frame_copy, mp_results)
            # cv2.imwrite(os.path.join(config.paths.keypoints, "..", "visualization", label, f"{idx}.jpg"), frame_copy)
            # del frame_copy

            # with transformation
            for transform in range(10):
                while True:
                    angle = np.random.randint(-15, 15)
                    tx = np.random.randint(-10, 10)
                    ty = np.random.randint(-10, 10)
                    scale_x = np.random.uniform(0.8, 1.2)
                    scale_y = np.random.uniform(0.8, 1.2)
                    frame = preprocess_image(frame, angle=angle, tx=tx, ty=ty, scale=(scale_x, scale_y))
                    landmarkers = init_landmarkers()
                    mp_results = mediapipe_detect_single(frame, landmarkers, 0)
                    keypoints = mediapipe_extract_single(mp_results)
                    # brute-force approach
                    # if all face-related keypoints (index: range(33, 125)) are 0, then the face is not detected
                    if np.all(keypoints[33:125] == 0):
                        del landmarkers
                        print(f"No face detected, retrying label {label} id {idx} transform {transform} ...")
                        continue
                    del landmarkers
                    break
                np.save(os.path.join(config.paths.keypoints, label, f"transform_{idx}_{transform}.npy"), keypoints)
                frame_copy = frame.copy()
                draw_landmarks(frame_copy, mp_results)
                cv2.imwrite(os.path.join(config.paths.keypoints, "..", "visualization", label, f"transform_{idx}_{transform}.jpg"), frame_copy)
                print(f"Random transformation for label {label} id {idx} transform {transform}")
                del frame_copy
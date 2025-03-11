import cv2
from mediapipe.python.solutions.holistic import Holistic
import numpy as np
import os
import argparse
from alive_progress import alive_bar
from image_transformation import *
import tensorflow as tf

if __name__ == "__main__":
    def mediapipe_detection(frame, holistic):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return result

    def extract_keypoints(result):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark]).flatten() if result.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in result.face_landmarks.landmark]).flatten() if result.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]).flatten() if result.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten() if result.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

    # supress warnings and errors that is not fatal to the program
    os.environ["GLOG_minloglevel"] ="3"

    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_dir", type=str, default=r"./dataset/tvb-hksl-news/frames")
    parser.add_argument("--keypoint_dir", type=str, default=r"./dataset/tvb-hksl-news/keypoints_mediapipe_transformed")
    parser.add_argument("--gpu", type=bool, default=True)
    parser.add_argument("--iteration", type=int, default=5)
    cur_frame_dir = parser.parse_args().frame_dir
    keypoint_dir = parser.parse_args().keypoint_dir
    GPU = parser.parse_args().gpu
    iteration = parser.parse_args().iteration

    if not GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    else:
        # reset the environment variable
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        physical_devices = tf.config.list_physical_devices('GPU')
        print(physical_devices)

    # frame_dir = r"E:\dataset\tvb-hksl-news\frames"
    # keypoint_dir = r"E:\dataset\tvb-hksl-news\keypoints_mediapipe"

    # a vanity progress bar
    total = 0
    date_list = os.listdir(cur_frame_dir)
    date_list.sort()
    for date in date_list:
        date_path = os.path.join(cur_frame_dir, date)
        name_list = os.listdir(date_path)
        total += len(name_list)
    print(f"Found {total} samples")
    total *= iteration # since we're doing it 10 times
    print(f"Total amount of keypoint files to generate: {total}")

    # TODO: rewrite this part to generate keypoints multiple times, each time applying a random transformation to the frame

    with Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, alive_bar(total, force_tty=True) as bar:
        date_list = os.listdir(cur_frame_dir)
        date_list.sort()
        for date in date_list:
            date_path = os.path.join(cur_frame_dir, date)
            name_list = os.listdir(date_path)
            name_list.sort()
            for name in name_list:
                for i in range(iteration):
                    save_dir = os.path.join(keypoint_dir, f"transformation_{i}", date)
                    save_name = f"{name}.npy"
                    if os.path.exists(os.path.join(save_dir, save_name)):
                        bar()
                        continue
                    os.makedirs(save_dir, exist_ok=True)
                    cur_frame_dir = os.path.join(date_path, name)
                    frame_list = os.listdir(cur_frame_dir)
                    frame_list.sort()

                    # define random transformation
                    angle = np.random.uniform(-10, 10)
                    tx = np.random.uniform(-50, 50)
                    ty = np.random.uniform(-50, 50)
                    scale = np.random.uniform(0.8, 1.2)
                    
                    keypoints_array = []
                    for frame in frame_list:
                        frame_path = os.path.join(cur_frame_dir, frame)
                        frame = cv2.imread(frame_path)
                        frame = pad_to_size(frame)
                        # for the first time, keep the original
                        if i != 0:
                            frame = apply_random_transformation(frame, angle, tx, ty, scale)
                            frame = ensure_in_bounds(frame)
                        result = mediapipe_detection(frame, holistic)
                        keypoints = extract_keypoints(result)
                        keypoints_array.append(keypoints)
                    keypoints_array = np.array(keypoints_array)
                    np.save(os.path.join(save_dir, save_name), keypoints_array)
                    bar()
import cv2
from mediapipe.python.solutions.holistic import Holistic
import numpy as np
import os
from alive_progress import alive_bar
import argparse

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

    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_dir", type=str, default=r"E:\dataset\tvb-hksl-news\frames")
    parser.add_argument("--keypoint_dir", type=str, default=r"E:\dataset\tvb-hksl-news\keypoints_mediapipe")
    frame_dir = parser.parse_args().frame_dir
    keypoint_dir = parser.parse_args().keypoint_dir

    # frame_dir = r"E:\dataset\tvb-hksl-news\frames"
    # keypoint_dir = r"E:\dataset\tvb-hksl-news\keypoints_mediapipe"

    # a vanity progress bar
    total = 0
    date_list = os.listdir(frame_dir)
    date_list.sort()
    for date in date_list:
        date_path = os.path.join(frame_dir, date)
        name_list = os.listdir(date_path)
        total += len(name_list)

    # TODO: rewrite this part to generate keypoints multiple times, each time applying a random transformation to the frame

    with Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, alive_bar(total, force_tty=True) as bar:
        # recursively walk through the frame directory
        # for directory frames/date/name/*.jpg, generate keypoints
        # then save the keypoints to keypoint_dir/date/name/*.npy
        # one npy file per name directory
        date_list = os.listdir(frame_dir)
        date_list.sort()
        for date in date_list:
            date_path = os.path.join(frame_dir, date)
            name_list = os.listdir(date_path)
            name_list.sort()
            for name in name_list:
                if os.path.exists(os.path.join(keypoint_dir, date, f"{name}.npy")):
                    # print(f"Skipping: {date}/{name}")
                    bar()
                    continue
                # DEBUG: print the current date and name
                print(f"Processing: {date}/{name}")
                name_path = os.path.join(date_path, name)
                keypoints_array = []
                frame_list = os.listdir(name_path)
                frame_list.sort()
                for frame in frame_list:
                    frame_path = os.path.join(name_path, frame)
                    frame = cv2.imread(frame_path)
                    result = mediapipe_detection(frame, holistic)
                    keypoints = extract_keypoints(result)
                    keypoints_array.append(keypoints)
                keypoints_array = np.array(keypoints_array)
                # DEBUG: print the shape of the keypoints array
                print(keypoints_array.shape)
                os.makedirs(os.path.join(keypoint_dir, date), exist_ok=True)
                np.save(os.path.join(keypoint_dir, date, f"{name}.npy"), keypoints_array)
                bar()
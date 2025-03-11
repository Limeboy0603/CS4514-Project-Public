import cv2
from mediapipe.python.solutions import drawing_utils
from mediapipe.python.solutions.holistic import Holistic
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS
import numpy as np
import os
import easygui
import time
import tensorflow as tf
from src_old.seq2seq.keypoint_preprocessing import get_mediapipe_keypoints_index

# input_path = r"E:\dataset\tvb-hksl-news\frames\2020-01-19\032266-032383"

def mediapipe_detection(frame, holistic):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = holistic.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return result

def draw_landmarks(frame, result):
    if result.pose_landmarks:
        drawing_utils.draw_landmarks(
            frame, result.face_landmarks, FACEMESH_TESSELATION
        )
        drawing_utils.draw_landmarks(
            frame, result.left_hand_landmarks, HAND_CONNECTIONS
        )
        drawing_utils.draw_landmarks(
            frame, result.right_hand_landmarks, HAND_CONNECTIONS
        )
        drawing_utils.draw_landmarks(
            frame, result.pose_landmarks, POSE_CONNECTIONS
        )

def extract_keypoints(result):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark]).flatten() if result.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in result.face_landmarks.landmark]).flatten() if result.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]).flatten() if result.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten() if result.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

if __name__ == "__main__":
    MODEL_PATH = "models/tvb_hksl_train_mediapipe_xyz.keras"

    model = tf.keras.models.load_model(MODEL_PATH)

    # Use EasyGUI to get the input path from the user
    input_path = easygui.diropenbox(title="Select a folder containing images to process", default=r"E:\dataset\tvb-hksl-news\frames\2020-01-19\032266-032383")

    if input_path is None:
        print("No folder selected. Exiting...")
        exit()

    input_path_images = os.listdir(input_path)
    input_path_images.sort()

    keypoints_array = []

    res = ""

    with Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for frame in input_path_images:
            image_path = os.path.join(input_path, frame)
            print("[DEBUG] Processing image:", image_path)
            frame = cv2.imread(image_path)
            # enlarge the frame proportionally
            frame = cv2.resize(frame, (frame.shape[1]*3, frame.shape[0]*3))
            result = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(result)
            keypoints = keypoints[get_mediapipe_keypoints_index()]
            keypoints_array.append(keypoints)
            res = model.predict(np.array(keypoints_array))
            cv2.putText(frame, res, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            draw_landmarks(frame, result)
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
    easygui.msgbox(res, title="Prediction Result")
    cv2.destroyAllWindows()
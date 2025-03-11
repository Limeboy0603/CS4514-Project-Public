import os
os.environ["GLOG_minloglevel"] ="3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from archive.cv_util import preprocess_image
import numpy as np
import cv2
from mediapipe.python.solutions import drawing_utils
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS, FACEMESH_NOSE
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python._framework_bindings.image import Image
from mediapipe.python._framework_bindings.image_frame import ImageFormat

"""
Feature selection vs Dimensionality reduction

We choose feature selection here because
- Face mesh has a lot of keypoints, and will differ from person to person. Thus, we only keep the important features.
- Anything below the shoulder is not important. No HKSL gesture involves the lower body.
"""

def get_mediapipe_keypoints_face_sublist() -> list[list[int]]:
    # reference: https://github.com/LearningnRunning/py_face_landmark_helper/blob/main/mediapipe_helper/config.py
    # image: https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    # related stack overflow post: https://stackoverflow.com/questions/74901522/can-mediapipe-specify-which-parts-of-the-face-mesh-are-the-lips-or-nose-or-eyes
    FACE_LIPS = [0, 267, 269, 270, 13, 14, 17, 402, 146, 405, 409, 415, 291, 37, 39, 40, 178, 308, 181, 310, 311, 312, 185, 314, 317, 318, 61, 191, 321, 324, 78, 80, 81, 82, 84, 87, 88, 91, 95, 375]
    LEFT_EYE = [384, 385, 386, 387, 388, 390, 263, 362, 398, 466, 373, 374, 249, 380, 381, 382]
    LEFT_EYEBROW = [293, 295, 296, 300, 334, 336, 276, 282, 283, 285]
    RIGHT_EYE = [160, 33, 161, 163, 133, 7, 173, 144, 145, 246, 153, 154, 155, 157, 158, 159]
    RIGHT_EYEBROW = [65, 66, 70, 105, 107, 46, 52, 53, 55, 63]
    FACE_NOSE = [1, 2, 4, 5, 6, 19, 275, 278, 294, 168, 45, 48, 440, 64, 195, 197, 326, 327, 344, 220, 94, 97, 98, 115]
    FACE_OVAL = [132, 389, 136, 10, 397, 400, 148, 149, 150, 21, 152, 284, 288, 162, 297, 172, 176, 54, 58, 323, 67, 454, 332, 338, 93, 356, 103, 361, 234, 109, 365, 379, 377, 378, 251, 127]

    target_list = [
        FACE_LIPS, 
        LEFT_EYE, 
        LEFT_EYEBROW, 
        RIGHT_EYE, 
        RIGHT_EYEBROW, 
        # FACE_OVAL
    ]
    return target_list

STATIC_FACE_KEYPOINT_INDEX = [i for sublist in get_mediapipe_keypoints_face_sublist() for i in sublist]

# filters out unnecessary keypoints
# this includes face, the 3 fingers of each hand, and anything below the shoulder
# reference: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#pose_landmarker_model
STATIC_POSE_KEYPOINT_INDEX = [i for i in range(11, 25) if i not in range(17, 23) and i not in [23, 24]]

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
pose_model_path = os.path.join(current_dir, "..", "mediapipe", "pose_landmarker_full.task")
face_model_path = os.path.join(current_dir, "..", "mediapipe", "face_landmarker.task")
hand_model_path = os.path.join(current_dir, "..", "mediapipe", "hand_landmarker.task")

pose_base_option = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path = pose_model_path,
        delegate=BaseOptions.Delegate.CPU
    ),
    running_mode=VisionTaskRunningMode.VIDEO,
)
face_base_option = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path = face_model_path,
        delegate=BaseOptions.Delegate.CPU
    ),
    running_mode=VisionTaskRunningMode.VIDEO,
)
hand_base_option = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path = hand_model_path,
        delegate=BaseOptions.Delegate.CPU,
    ),
    running_mode=VisionTaskRunningMode.VIDEO,
    num_hands=2,
)

def init_landmarkers():
    pose_landmarker = PoseLandmarker.create_from_options(pose_base_option)
    face_landmarker = FaceLandmarker.create_from_options(face_base_option)
    hand_landmarker = HandLandmarker.create_from_options(hand_base_option)
    return pose_landmarker, face_landmarker, hand_landmarker

def mediapipe_detect_single(frame, landmarkers, timestamp):
    pose_landmarker, face_landmarker, hand_landmarker = landmarkers
    pose_result = pose_landmarker.detect_for_video(Image(data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), image_format=ImageFormat.SRGB), timestamp)
    face_result = face_landmarker.detect_for_video(Image(data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), image_format=ImageFormat.SRGB), timestamp)
    hand_result = hand_landmarker.detect_for_video(Image(data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), image_format=ImageFormat.SRGB), timestamp)
    return pose_result, face_result, hand_result

def mediapipe_extract_single(results):
    pose_result, face_result, hand_result = results

    # take pose from the result
    pose_landmarks = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
    # pose = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks]) if pose_landmarks else np.zeros((33, 3))
    pose_unfiltered = [[lm.x, lm.y, lm.z] for lm in pose_landmarks] if pose_landmarks else None
    pose = np.array([pose_unfiltered[i] for i in STATIC_POSE_KEYPOINT_INDEX]) if pose_unfiltered else np.zeros((len(STATIC_POSE_KEYPOINT_INDEX), 3))

    # take the first face from the result
    face_landmarks = face_result.face_landmarks[0] if face_result.face_landmarks else None
    face_unfiltered = [[lm.x, lm.y, lm.z] for lm in face_landmarks] if face_landmarks else None
    face = np.array([face_unfiltered[i] for i in STATIC_FACE_KEYPOINT_INDEX]) if face_unfiltered else np.zeros((len(STATIC_FACE_KEYPOINT_INDEX), 3))

    # take both hands
    hand_landmarks = hand_result.hand_landmarks # list of 0 to 2 hands
    hand_handness = hand_result.handedness
    left_hand = np.zeros((21, 3))
    right_hand = np.zeros((21, 3))
    for i, hand in enumerate(hand_handness):
        if hand[0].category_name == "Left":
            left_hand = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks[i]])
        elif hand[0].category_name == "Right":
            right_hand = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks[i]])

    concat = np.concatenate([pose, face, left_hand, right_hand])
    return concat

def mediapipe_detect_multiple(frames, landmarkers, show_image=False): # frames is a list of frames
    pose_landmarker, face_landmarker, hand_landmarker = landmarkers
    pose_results = []
    face_results = []
    hand_results = []
    for i in range(len(frames)):
        result = mediapipe_detect_single(frames[i], landmarkers, i)
        pose_results.append(result[0])
        face_results.append(result[1])
        hand_results.append(result[2])
        if show_image:
            draw_landmarks(frames[i], result)
            cv2.imshow("frame", frames[i])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    return pose_results, face_results, hand_results

def mediapipe_extract_multiple(results):
    ret = []
    for result in zip(*results):
        ret.append(mediapipe_extract_single(result))
    return np.array(ret)

def draw_landmarks(frame, results) -> None:
    pose_result, face_result, hand_result = results
    drawing_spec = drawing_utils.DrawingSpec(
        color=(0, 255, 0), thickness=1, circle_radius=1
    )

    if pose_result.pose_landmarks:
        # pose_landmark_proto = landmark_pb2.NormalizedLandmarkList()
        # pose_landmark_proto.landmark.extend([
        #     landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in pose_result.pose_landmarks[0]
        # ])
        # filtered_pose_connections = [(i, j) for i, j in POSE_CONNECTIONS if i in STATIC_POSE_KEYPOINT_INDEX and j in STATIC_POSE_KEYPOINT_INDEX]
        # drawing_utils.draw_landmarks(
        #     frame, pose_landmark_proto, filtered_pose_connections, drawing_spec, drawing_spec
        # )
        pose_landmarks = pose_result.pose_landmarks[0]
        for i in STATIC_POSE_KEYPOINT_INDEX:
            lm = pose_landmarks[i]
            cv2.circle(
                frame, 
                (
                    int(lm.x * frame.shape[1]),
                    int(lm.y * frame.shape[0])
                ), 
                drawing_spec.circle_radius, 
                drawing_spec.color, 
                drawing_spec.thickness
            )
        for connection in POSE_CONNECTIONS:
            if connection[0] not in STATIC_POSE_KEYPOINT_INDEX or connection[1] not in STATIC_POSE_KEYPOINT_INDEX:
                continue
            start = pose_landmarks[connection[0]]
            end = pose_landmarks[connection[1]]
            cv2.line(
                frame,
                (int(start.x * frame.shape[1]), int(start.y * frame.shape[0])),
                (int(end.x * frame.shape[1]), int(end.y * frame.shape[0])),
                drawing_spec.color,
                drawing_spec.thickness,
            )

    if hand_result.hand_landmarks:
        for hand_landmark in hand_result.hand_landmarks:
            hand_landmark_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmark_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmark
            ])
            drawing_utils.draw_landmarks(
                frame, hand_landmark_proto, HAND_CONNECTIONS, drawing_spec, drawing_spec
            )

    if face_result.face_landmarks:
        face_landmarks = face_result.face_landmarks[0]
        for i in STATIC_FACE_KEYPOINT_INDEX:
            lm = face_landmarks[i]
            cv2.circle(
                frame, 
                (
                    int(lm.x * frame.shape[1]),
                    int(lm.y * frame.shape[0])
                ), 
                drawing_spec.circle_radius, 
                drawing_spec.color, 
                drawing_spec.thickness
            )

        for connection in FACEMESH_CONTOURS:
            if connection[0] not in STATIC_FACE_KEYPOINT_INDEX or connection[1] not in STATIC_FACE_KEYPOINT_INDEX:
                continue
            start = face_landmarks[connection[0]]
            end = face_landmarks[connection[1]]
            cv2.line(
                frame,
                (int(start.x * frame.shape[1]), int(start.y * frame.shape[0])),
                (int(end.x * frame.shape[1]), int(end.y * frame.shape[0])),
                drawing_spec.color,
                drawing_spec.thickness,
            )

def preprocess_keypoints(keypoints, angle=0, tx=0, ty=0, tz=0, scale=1) -> np.ndarray:
    # given a list of 3D keypoints, manually compute the transformation
    # angle: in degrees
    # tx, ty: translation in pixels
    # tz: translation in z-axis
    # scale: scaling factor
    keypoints = keypoints.copy()
    # apply rotation
    angle = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    keypoints = np.dot(rotation_matrix, keypoints.T).T
    # apply translation
    keypoints[:, 0] += tx
    keypoints[:, 1] += ty
    keypoints[:, 2] += tz
    # apply scaling
    # keypoints[:, :2] *= scale
    # scale according to position, where 0.5 is the center
    keypoints[:, :2] = scale * (keypoints[:, :2] - 0.5) + 0.5
    return keypoints

def preprocess_keypoints_multiple(keypoints, angle=0, tx=0, ty=0, tz=0, scale=1) -> list[np.ndarray]:
    keypoints = keypoints.copy()
    # print(keypoints[0].shape)
    for i in range(len(keypoints)):
        keypoints[i] = preprocess_keypoints(keypoints[i], angle, tx, ty, tz, scale)
    return keypoints

# def landmark_diff(keypoints1, keypoints2) -> np.ndarray:
#     return keypoints1 - keypoints2

# def has_large_change(keypoints1, keypoints2) -> bool:
#     diff = landmark_diff(keypoints1, keypoints2)
#     pose_diff = diff[:33]
#     face_diff = diff[33:33+len(STATIC_FACE_KEYPOINT_INDEX)]
#     left_hand_diff = diff[33+len(STATIC_FACE_KEYPOINT_INDEX):33+len(STATIC_FACE_KEYPOINT_INDEX)+63]
#     right_hand_diff = diff[33+len(STATIC_FACE_KEYPOINT_INDEX)+63:]

#     pose_large_change = np.linalg.norm(pose_diff) > 0.1
#     face_large_change = np.linalg.norm(face_diff) > 0.1
#     left_hand_large_change = np.linalg.norm(left_hand_diff) > 0.1
#     right_hand_large_change = np.linalg.norm(right_hand_diff) > 0.1
#     return pose_large_change or face_large_change or left_hand_large_change or right_hand_large_change

def get_flattened_index_by_part():
    return {
        "pose": list(range(33)),
        "face": list(range(33, 33+len(STATIC_FACE_KEYPOINT_INDEX))),
        "left_hand": list(range(33+len(STATIC_FACE_KEYPOINT_INDEX), 33+len(STATIC_FACE_KEYPOINT_INDEX)+63)),
        "right_hand": list(range(33+len(STATIC_FACE_KEYPOINT_INDEX)+63, 33+len(STATIC_FACE_KEYPOINT_INDEX)+63+63))
    }

def apply_weighting_to_flattened(keypoint):
    parts = get_flattened_index_by_part()
    keypoint = keypoint.copy()
    keypoint[parts["left_hand"]] *= 1
    keypoint[parts["right_hand"]] *= 1
    keypoint[parts["pose"]] *= 1
    return keypoint

if __name__ == "__main__":
    print(len(STATIC_POSE_KEYPOINT_INDEX), len(STATIC_FACE_KEYPOINT_INDEX))
    # print range of each part
    print(0, 33)
    print(33, 33+len(STATIC_FACE_KEYPOINT_INDEX))
    print(33+len(STATIC_FACE_KEYPOINT_INDEX), 33+len(STATIC_FACE_KEYPOINT_INDEX)+63)
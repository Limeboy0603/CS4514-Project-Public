from mediapipe.python.solutions.holistic import Holistic
from mediapipe.python.solutions import drawing_utils
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS
import cv2 
import numpy as np

def get_mediapipe_keypoints_face_sublist() -> list[list[int]]:
    # face
    # NOTE: the following keypoint indices are HARD-CODED based on the visualization of the face mesh
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
    return [
        FACE_LIPS, 
        LEFT_EYE, 
        LEFT_EYEBROW, 
        RIGHT_EYE, 
        RIGHT_EYEBROW, 
        # FACE_NOSE, 
        # FACE_OVAL
    ]

STATIC_FACE_KEYPOINT_INDEX = [i for sublist in get_mediapipe_keypoints_face_sublist() for i in sublist]

def get_mediapipe_keypoints_index() -> list[int]:
    POSE_UNPROCESSED = range(0, 33*4)
    # POSE = [i for i in POSE_UNPROCESSED if i % 4 != 3]
    # for x, y only
    # discard Z due to documentation https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/holistic.md
    POSE = [i for i in POSE_UNPROCESSED if i % 4 != 2 and i % 4 != 3]

    # face keypoints are in x, y, z format flattened, so we need to capture all x, y, z values
    static_face_keypoint_processed = [i*3 + j for i in STATIC_FACE_KEYPOINT_INDEX for j in range(3)]
    # for x, y only
    # static_face_keypoint_processed = [i*3 + j for i in STATIC_FACE_KEYPOINT_INDEX for j in range(2)]
    FACE = [33*4 + i for i in static_face_keypoint_processed]

    # hands
    LEFT_HAND = list(range(33*4 + 468*3, 33*4 + 468*3 + 21*3))
    RIGHT_HAND = list(range(33*4 + 468*3 + 21*3, 33*4 + 468*3 + 21*3 + 21*3))
    # for x, y only
    # LEFT_HAND = [i for i in list(range(33*4 + 468*3, 33*4 + 468*3 + 21*3)) if i % 3 != 2]
    # RIGHT_HAND = [i for i in list(range(33*4 + 468*3 + 21*3, 33*4 + 468*3 + 21*3 + 21*3)) if i % 3 != 2]
    KEYPOINTS_INDEX = POSE + FACE + LEFT_HAND + RIGHT_HAND
    return KEYPOINTS_INDEX, [POSE, FACE, LEFT_HAND, RIGHT_HAND]

STATIC_KEYPOINTS_INDEX = get_mediapipe_keypoints_index()[0] # saves time by not recalculating the indices

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
    # print(pose.shape, face.shape, lh.shape, rh.shape)
    concat = np.concatenate([pose, face, lh, rh])
    # print(concat.shape)
    # print(STATIC_KEYPOINTS_INDEX)
    return concat[STATIC_KEYPOINTS_INDEX]

def draw_landmarks(frame, result):
    # DEBUG: list all attributes of result
    # print(dir(result))
    # print(dir(result.face_landmarks))
    drawing_spec = drawing_utils.DrawingSpec(
        color=(0, 255, 0), thickness=1, circle_radius=1
    )
    drawing_utils.draw_landmarks(
        frame, result.pose_landmarks, POSE_CONNECTIONS, drawing_spec, drawing_spec
    )
    drawing_utils.draw_landmarks(
        frame, result.face_landmarks, FACEMESH_CONTOURS, drawing_spec, drawing_spec
    )
    drawing_utils.draw_landmarks(
        frame, result.left_hand_landmarks, HAND_CONNECTIONS, drawing_spec, drawing_spec
    )
    drawing_utils.draw_landmarks(
        frame, result.right_hand_landmarks, HAND_CONNECTIONS, drawing_spec, drawing_spec
    )

def draw_landmarks_kp(frame, keypoints):
    pose_keypoints, face_keypoints, left_hand_keypoints, right_hand_keypoints = get_raw_coords(keypoints)
    width, height = frame.shape[1], frame.shape[0]
    for x, y in pose_keypoints:
        cv2.circle(frame, (int(x * width), int(y * height)), 3, (0, 255, 0), -1)
    for x, y, _ in face_keypoints:
        cv2.circle(frame, (int(x * width), int(y * height)), 3, (255, 0, 0), -1)
    for x, y, _ in left_hand_keypoints:
        cv2.circle(frame, (int(x * width), int(y * height)), 3, (0, 0, 255), -1)
    for x, y, _ in right_hand_keypoints:
        cv2.circle(frame, (int(x * width), int(y * height)), 3, (0, 0, 255), -1)

def mediapipe_extract_single(frame, holistic, visualize=False):
    result = mediapipe_detection(frame, holistic)
    keypoints = extract_keypoints(result)
    if visualize:
        # draw_landmarks(frame, result)
        draw_landmarks_kp(frame, keypoints)
        # cv2.imshow("frame", frame)
        cv2.waitKey(1)
    return keypoints

def mediapipe_extract_multiple(frames, visualize=False):
    with Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.3) as holistic:
        keypoints = []
        for frame in frames:
            keypoint = mediapipe_extract_single(frame, holistic, visualize)
            keypoints.append(keypoint)
        return np.array(keypoints)
    
def preprocess_keypoints(keypoints, angle=0, tx=0, ty=0, tz=0, scale=1):
    POSE, FACE, LEFT_HAND, RIGHT_HAND = get_mediapipe_keypoints_index()[1]
    pose_indices = range(len(POSE))
    face_indices = range(len(POSE), len(POSE) + len(FACE))
    left_hand_indices = range(len(POSE) + len(FACE), len(POSE) + len(FACE) + len(LEFT_HAND))
    right_hand_indices = range(len(POSE) + len(FACE) + len(LEFT_HAND), len(POSE) + len(FACE) + len(LEFT_HAND) + len(RIGHT_HAND))
    # print(POSE, FACE, LEFT_HAND, RIGHT_HAND)
    keypoints = keypoints.copy()
    # print(keypoints.shape)
    pose_keypoints = keypoints[pose_indices]
    face_keypoints = keypoints[face_indices]
    left_hand_keypoints = keypoints[left_hand_indices]
    right_hand_keypoints = keypoints[right_hand_indices]

    angle = np.radians(angle)

    # pose only has X and Y in flattened format
    pose_keypoints = pose_keypoints.reshape(-1, 2)
    # rotate each point by angle
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)], 
        [np.sin(angle), np.cos(angle)]
    ])
    pose_keypoints = np.dot(pose_keypoints, rotation_matrix)
    pose_keypoints[:, 0] += tx
    pose_keypoints[:, 1] += ty
    pose_keypoints[:, :2] = scale * (pose_keypoints[:, :2] - 0.5) + 0.5
    keypoints[pose_indices] = pose_keypoints.flatten()

    # other parts have X, Y, Z in flattened format
    face_keypoints = face_keypoints.reshape(-1, 3)
    left_hand_keypoints = left_hand_keypoints.reshape(-1, 3)
    right_hand_keypoints = right_hand_keypoints.reshape(-1, 3)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    face_keypoints = np.dot(face_keypoints, rotation_matrix)
    left_hand_keypoints = np.dot(left_hand_keypoints, rotation_matrix)
    right_hand_keypoints = np.dot(right_hand_keypoints, rotation_matrix)
    face_keypoints[:, 0] += tx
    face_keypoints[:, 1] += ty
    face_keypoints[:, 2] += tz
    left_hand_keypoints[:, 0] += tx
    left_hand_keypoints[:, 1] += ty
    left_hand_keypoints[:, 2] += tz
    right_hand_keypoints[:, 0] += tx
    right_hand_keypoints[:, 1] += ty
    right_hand_keypoints[:, 2] += tz
    face_keypoints[:, :2] = scale * (face_keypoints[:, :2] - 0.5) + 0.5
    left_hand_keypoints[:, :2] = scale * (left_hand_keypoints[:, :2] - 0.5) + 0.5
    right_hand_keypoints[:, :2] = scale * (right_hand_keypoints[:, :2] - 0.5) + 0.5
    # NOTE: Z needs to be scaled. see output of experiments_kp_Z_test.py
    face_keypoints[:, 2] = scale * face_keypoints[:, 2]
    left_hand_keypoints[:, 2] = scale * left_hand_keypoints[:, 2]
    right_hand_keypoints[:, 2] = scale * right_hand_keypoints[:, 2]
    keypoints[face_indices] = face_keypoints.flatten()
    keypoints[left_hand_indices] = left_hand_keypoints.flatten()
    keypoints[right_hand_indices] = right_hand_keypoints.flatten()
    return keypoints

def preprocess_keypoints_multiple(keypoints, angle=0, tx=0, ty=0, tz=0, scale=1):
    return np.array([preprocess_keypoints(keypoint, angle, tx, ty, tz, scale) for keypoint in keypoints])

def create_holistic():
    return Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.3)

def get_raw_coords(keypoints):
    POSE, FACE, LEFT_HAND, RIGHT_HAND = get_mediapipe_keypoints_index()[1]
    pose_indices = range(len(POSE))
    face_indices = range(len(POSE), len(POSE) + len(FACE))
    left_hand_indices = range(len(POSE) + len(FACE), len(POSE) + len(FACE) + len(LEFT_HAND))
    right_hand_indices = range(len(POSE) + len(FACE) + len(LEFT_HAND), len(POSE) + len(FACE) + len(LEFT_HAND) + len(RIGHT_HAND))
    keypoints = keypoints.copy()
    pose_keypoints = keypoints[pose_indices]
    face_keypoints = keypoints[face_indices]
    left_hand_keypoints = keypoints[left_hand_indices]
    right_hand_keypoints = keypoints[right_hand_indices]
    pose_keypoints = pose_keypoints.reshape(-1, 2)
    face_keypoints = face_keypoints.reshape(-1, 3)
    left_hand_keypoints = left_hand_keypoints.reshape(-1, 3)
    right_hand_keypoints = right_hand_keypoints.reshape(-1, 3)
    return pose_keypoints, face_keypoints, left_hand_keypoints, right_hand_keypoints
import argparse
import keras
from mediapipe.python.solutions import drawing_utils
from mediapipe.python.solutions.holistic import Holistic
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS, FACEMESH_NOSE
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS
import numpy as np
import cv2
import os
import easygui
import json
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--use_gui", action="store_true", help="Use GUI for file selection")
parser.add_argument("--encoder_path", type=str, help="Path to the encoder model. Default: 'model/train_LSTM_weighted_32_512_1024_1/encoder.keras'", default="model/train_LSTM_weighted_32_512_1024_1/encoder.keras")
parser.add_argument("--decoder_path", type=str, help="Path to the decoder model. Default: 'model/train_LSTM_weighted_32_512_1024_1/decoder.keras'", default="model/train_LSTM_weighted_32_512_1024_1/decoder.keras")
parser.add_argument("--input_path", type=str, help="Path to the folder containing frames. Default: 'dataset/tvb-hksl-news/frames/2020-03-14/002832-003033'", default="dataset/tvb-hksl-news/frames/2020-03-14/002832-003033")
parser.add_argument("--random_angle", type=float, help="Random angle for image transformation. Default: 0", default=0)
parser.add_argument("--random_tx", type=float, help="Random translation in x-axis for image transformation. Default: 0", default=0)
parser.add_argument("--random_ty", type=float, help="Random translation in y-axis for image transformation. Default: 0", default=0)
parser.add_argument("--random_scale", type=float, help="Random scaling for image transformation. Default: 1", default=1)

def pad_to_size(image, target_size=(1920, 1080)):
    rows, cols, _ = image.shape
    target_cols, target_rows = target_size
    scale_factor = min(target_cols / cols, target_rows / rows)
    new_cols = int(cols * scale_factor)
    new_rows = int(rows * scale_factor)
    resized_image = cv2.resize(image, (new_cols, new_rows), interpolation=cv2.INTER_LINEAR)

    top = (target_rows - new_rows) // 2
    bottom = target_rows - new_rows - top
    left = (target_cols - new_cols) // 2
    right = target_cols - new_cols - left
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded_image

def apply_random_transformation(image, angle, tx, ty, scale):
    rows, cols, _ = image.shape

    # rotation
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))

    # translation
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(rotated_image, M, (cols, rows))

    # scaling
    scaled_image = cv2.resize(translated_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # consistency
    final_image = cv2.resize(scaled_image, (cols, rows), interpolation=cv2.INTER_LINEAR)

    return final_image

def ensure_in_bounds(image, target_size=(1920, 1080)):
    rows, cols, _ = image.shape
    target_cols, target_rows = target_size

    # create a black canvas of the target size
    canvas = np.zeros((target_rows, target_cols, 3), dtype=np.uint8)

    # find the bounding box of non-black pixels
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])

    top = (target_rows - h) // 2
    left = (target_cols - w) // 2
    canvas[top:top+h, left:left+w] = image[y:y+h, x:x+w]

    return canvas

def preprocess_image(image, target_size=(1920, 1080), angle=0, tx=0, ty=0, scale=1):
    padded_image = pad_to_size(image, target_size)
    transformed_image = apply_random_transformation(padded_image, angle, tx, ty, scale)
    # final_image = ensure_in_bounds(transformed_image, target_size)
    return transformed_image

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
    return [FACE_LIPS, LEFT_EYE, LEFT_EYEBROW, RIGHT_EYE, RIGHT_EYEBROW, FACE_NOSE, FACE_OVAL]

STATIC_FACE_KEYPOINT_INDEX = [i for sublist in get_mediapipe_keypoints_face_sublist() for i in sublist]

def get_mediapipe_keypoints_index() -> list[int]:
    POSE_UNPROCESSED = range(0, 33*4)
    # POSE = [i for i in POSE_UNPROCESSED if i % 4 != 3]
    # for x, y only
    # discard Z due to documentation https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/holistic.md
    POSE = [i for i in POSE_UNPROCESSED if i % 4 != 2 and i % 4 != 3]

    FACE_UNPROCESSED = [i + 33*4 for i in STATIC_FACE_KEYPOINT_INDEX]
    # face keypoints are in x, y, z format flattened, so we need to capture all x, y, z values
    FACE = [i for j in range(0, len(FACE_UNPROCESSED), 3) for i in range(FACE_UNPROCESSED[j], FACE_UNPROCESSED[j] + 3)]
    # for x, y only
    # FACE = [i for j in range(0, len(FACE_UNPROCESSED), 3) for i in range(FACE_UNPROCESSED[j], FACE_UNPROCESSED[j] + 2)]

    # hands
    LEFT_HAND = list(range(33*4 + 468*3, 33*4 + 468*3 + 21*3))
    RIGHT_HAND = list(range(33*4 + 468*3 + 21*3, 33*4 + 468*3 + 21*3 + 21*3))
    # for x, y only
    # LEFT_HAND = [i for i in list(range(33*4 + 468*3, 33*4 + 468*3 + 21*3)) if i % 3 != 2]
    # RIGHT_HAND = [i for i in list(range(33*4 + 468*3 + 21*3, 33*4 + 468*3 + 21*3 + 21*3)) if i % 3 != 2]
    KEYPOINTS_INDEX = POSE + FACE + LEFT_HAND + RIGHT_HAND
    return KEYPOINTS_INDEX

STATIC_KEYPOINTS_INDEX = get_mediapipe_keypoints_index() # saves time by not recalculating the indices

def mediapipe_detection(frame, holistic):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = holistic.process(frame)
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return result

def mediapipe_extract_keypoints(result):
    # print(len(result.pose_landmarks.landmark))
    # print(len(result.face_landmarks.landmark))
    # print(len(result.left_hand_landmarks.landmark))
    # print(len(result.right_hand_landmarks.landmark))
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark]).flatten() if result.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in result.face_landmarks.landmark]).flatten() if result.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]).flatten() if result.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten() if result.right_hand_landmarks else np.zeros(21*3)
    concat = np.concatenate([pose, face, lh, rh])
    return concat[STATIC_KEYPOINTS_INDEX]

# def mediapipe_extract(frames): # frames is a list of frames
#     with Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2) as holistic:
#         return [mediapipe_extract_keypoints(mediapipe_detection(frame, holistic)) for frame in frames]

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
    # drawing_utils.draw_landmarks(
    #     frame, result.face_landmarks, frozenset().union(*[FACEMESH_CONTOURS, FACEMESH_NOSE]), drawing_spec, drawing_spec
    # )
    drawing_utils.draw_landmarks(
        frame, result.left_hand_landmarks, HAND_CONNECTIONS, drawing_spec, drawing_spec
    )
    drawing_utils.draw_landmarks(
        frame, result.right_hand_landmarks, HAND_CONNECTIONS, drawing_spec, drawing_spec
    )

    # manually draw the face mesh using STATIC_FACE_KEYPOINT_INDEX
    face_landmarks = result.face_landmarks.landmark
    for i in range(len(STATIC_FACE_KEYPOINT_INDEX)):
        lm = face_landmarks[STATIC_FACE_KEYPOINT_INDEX[i]]
        x = int(lm.x * frame.shape[1])
        y = int(lm.y * frame.shape[0])
        cv2.circle(frame, (x, y), drawing_spec.circle_radius, drawing_spec.color, drawing_spec.thickness)

    for connection in frozenset().union(*[FACEMESH_CONTOURS, FACEMESH_NOSE]):
        start_idx, end_idx = connection
        start_lm = face_landmarks[start_idx]
        end_lm = face_landmarks[end_idx]
        start_x = int(start_lm.x * frame.shape[1])
        start_y = int(start_lm.y * frame.shape[0])
        end_x = int(end_lm.x * frame.shape[1])
        end_y = int(end_lm.y * frame.shape[0])
        cv2.line(frame, (start_x, start_y), (end_x, end_y), drawing_spec.color, drawing_spec.thickness)

if __name__ == "__main__":
    args = parser.parse_args()

    USE_GUI = args.use_gui
    if USE_GUI:
        encoder_path = easygui.fileopenbox(title="Select the encoder model", default="model/train_LSTM_weighted_32_512_1024_1/encoder.keras")
        decoder_path = easygui.fileopenbox(title="Select the decoder model", default="model/train_LSTM_weighted_32_512_1024_1/decoder.keras")
        input_path = easygui.diropenbox(title="Select a folder containing images to process", default="dataset/tvb-hksl-news/frames/2020-03-14/002832-003033")
    else:
        # encoder_path = "model/train_encoder_weighted_32_512_1024_1.keras"
        # decoder_path = "model/train_decoder_weighted_32_512_1024_1.keras"
        # input_path = "dataset/tvb-hksl-news/frames/2020-01-16/016193-016336"
        encoder_path = args.encoder_path
        decoder_path = args.decoder_path
        input_path = args.input_path
    encoder_model = keras.models.load_model(encoder_path)
    decoder_model = keras.models.load_model(decoder_path)

    # disable cudnn
    for layer in encoder_model.layers:
        if isinstance(layer, keras.layers.LSTM):
            layer.use_cudnn = False
    for layer in decoder_model.layers:
        if isinstance(layer, keras.layers.LSTM):
            layer.use_cudnn = False

    input_path_images = os.listdir(input_path)
    input_path_images.sort()
    
    frames = [cv2.imread(os.path.join(input_path, frame)) for frame in input_path_images]

    if USE_GUI:
        use_random = easygui.ynbox("Use random transformation?", "Random Transformation", ["Yes", "No"])
    else: # for testing purposes
        use_random = False
    if use_random:
        angle = np.random.uniform(-10, 10)
        tx = np.random.uniform(-100, 100)
        ty = np.random.uniform(-60, 60)
        scale = np.random.uniform(0.8, 1.2)
    else:
        angle = args.random_angle
        tx = args.random_tx
        ty = args.random_ty
        scale = args.random_scale
    frames = [preprocess_image(frame, angle=angle, tx=tx, ty=ty, scale=scale) for frame in frames]
    # keypoints = mediapipe_extract(frames)

    with open("data/word_dict.json", "r") as f:
        word_dict = json.load(f)
    with open("data/reverse_word_dict.json", "r") as f:
        reverse_word_dict = json.load(f)
    reverse_word_dict = {int(k): v for k, v in reverse_word_dict.items()}

    def decode_sequence(input_seq):
        states_value = encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, len(word_dict)))
        target_seq[0, 0, word_dict["<START>"]] = 1
        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = reverse_word_dict[sampled_token_index]
            decoded_sentence.append(sampled_word)
            if sampled_word == "<END>" or len(decoded_sentence) > 47: # hard-coded: the longest supported sentence length
                stop_condition = True
            target_seq = np.zeros((1, 1, len(word_dict)))
            target_seq[0, 0, sampled_token_index] = 1
            states_value = [h, c]
        return decoded_sentence

    keypoints_seq = []

    with Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0) as holistic:
        for frame in frames:
            result = mediapipe_detection(frame, holistic)
            keypoints = mediapipe_extract_keypoints(result)
            keypoints_seq.append(keypoints)
            if USE_GUI:
                draw_landmarks(frame, result)
                cv2.imshow("Frame", frame)
                cv2.waitKey(1)

    keypoints_seq_np = np.array(keypoints_seq)
    keypoints_seq_np = keras.preprocessing.sequence.pad_sequences([keypoints_seq_np], maxlen=376, padding="post", dtype="float32")
    print(keypoints_seq_np.shape)

    def generate_sentence(sentence):
        post_obj = {"prompt": sentence}
        # response = requests.post("172.17.0.1:5000", json=post_obj)
        response = requests.post("http://127.0.0.1:5001/combine", json=post_obj)
        return response.json()["response"]

    pred = decode_sequence(keypoints_seq_np)
    pred_processed = [word for word in pred if word[0] != "<" and word[-1] != ">"]
    res_str = f"""
    Original: {" ".join(pred)}
    Processed: {" ".join(pred_processed)}
    Combined: {generate_sentence(" ".join(pred_processed))}
    """
    if USE_GUI:
        easygui.msgbox(res_str, title="Prediction Result")
        cv2.destroyAllWindows()
    else:
        print(res_str)
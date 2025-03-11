import cv2
import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'torch'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import keras
from llm_util import slt_sentence_predict

from config import config_parser, Config
# import mp_util
from mp_util_legacy import create_holistic, mediapipe_extract_single

def main(config_path: str, source_video: str = None):
    # config = config_parser("config/config_clip.yaml")
    config: Config = config_parser(config_path)
    # 0 for camera
    # 1 for OBS virtual camera

    # if source_video is provided, use the video file instead of camera
    if source_video is None: # uses camera
        cap = cv2.VideoCapture(config.capture.source)
        width = int(config.capture.resolution_width)
        height = int(config.capture.resolution_height)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    else: # uses video
        cap = cv2.VideoCapture(source_video)

    sentence = []
    predictions = []

    slr_model = keras.models.load_model(config.paths.model)
    holistic = create_holistic()

    all_labels = sorted(os.listdir(config.paths.keypoints))
    seq_max_len = config.sequence.frame

    print(all_labels)
    print(seq_max_len)
    # exit()

    # frames = []
    keypoints = []
    frame_num = 0

    pred_text = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # frames.append(frame)
        frame_num += 1
        keypoint = mediapipe_extract_single(frame, holistic, visualize=True)
        keypoint = keypoint.flatten()
        keypoints.append(keypoint)
        cv2.putText(frame, "Prediction: " + (predictions[-1] if predictions else ""), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("frame", frame)
        # if len(frames) % 30 == 0:
        if frame_num % 30 == 0:
            keypoints_group = keypoints[-30:]
            keypoints_group = np.array([keypoints_group])
            res = slr_model.predict(keypoints_group)
            res_label = all_labels[np.argmax(res)]
            predictions.append(res_label)
            if len(sentence) == 0 or res_label != sentence[-1]:
                sentence.append(res_label)
            print(res_label)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    del holistic
    del cap

    print(sentence)
    print(predictions)
    print(slt_sentence_predict(" ".join(sentence), config))

if __name__ == "__main__":
    main("config/config_clip.yaml" if os.name == "nt" else "config/config_clip_linux.yaml")
import cv2
import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'torch'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import keras
from llm_util import slt_sentence_predict

from config import config_parser
# import mp_util
from mp_util_legacy import create_holistic, mediapipe_extract_single

def main(config_path: str, source_video: str = None):
    config = config_parser(config_path)

    # if source_video is provided, use the video file instead of camera
    if source_video is None: # uses camera
        cap = cv2.VideoCapture(config.capture.source)
        width = int(config.capture.resolution_width)
        height = int(config.capture.resolution_height)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    else: # uses video
        cap = cv2.VideoCapture(source_video)

    sequence = []
    sentence = []
    sentence_no_threshold = []
    predictions = []
    threshold = 0.9

    slr_model = keras.models.load_model(config.paths.model)

    # landmarkers = mp_util.init_landmarkers()
    holistic = create_holistic()
    frame_num = 0

    all_labels = sorted(os.listdir(config.paths.keypoints))
    seq_max_len = config.sequence.frame

    print(all_labels)
    print(seq_max_len)
    # exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        keypoints = mediapipe_extract_single(frame, holistic, visualize=True)
        keypoints = keypoints.flatten()
        sequence.append(keypoints)
        sequence = sequence[-seq_max_len:]
        if len(sequence) == seq_max_len:
            sequence_np = np.array([np.array(sequence)])
            # print(sequence_np.shape)
            res = slr_model.predict(sequence_np)[0]
            print(res)
            res_score = res[np.argmax(res)]
            res_label = all_labels[np.argmax(res)]
            print(res_score, res_label)

            predictions.append(res_label)
            if res_score > threshold:
                if len(sentence) == 0 or sentence[-1] != res_label:
                    sentence.append(res_label)
            if len(sentence_no_threshold) == 0 or sentence_no_threshold[-1] != res_label:
                sentence_no_threshold.append(res_label)
            for i, probability in enumerate(res):
                cv2.putText(frame, f"{all_labels[i]}: {int(probability*100)}%", (20, 100+i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, " ".join(sentence), (20, 100+len(all_labels)*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    del holistic
    del cap
    
    # print the sentence
    sentence = [word for word in sentence if word != "#BLANK"]
    print(predictions)
    print(" ".join(sentence))
    print(" ".join(sentence_no_threshold))
    print(slt_sentence_predict(" ".join(sentence), config))
    print(slt_sentence_predict(" ".join(sentence_no_threshold), config))

if __name__ == "__main__":
    main("config/config_clip.yaml" if os.name == "nt" else "config/config_clip_linux.yaml")
import cv2
import numpy as np
import keras
from config import config_parser
from mp_util import init_landmarkers, mediapipe_detect_single, mediapipe_extract_single, draw_landmarks, apply_weighting_to_flattened

if __name__ == "__main__":
    config = config_parser("config/config_image.yaml")

    capture_source = config.capture.source
    cap = cv2.VideoCapture(capture_source)
    width = int(config.capture.resolution_width)
    height = int(config.capture.resolution_height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    sentence = []
    predictions = []

    dictionary = config.dictionary

    model = keras.models.load_model(config.paths.model)

    landmarkers = init_landmarkers()
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        results = mediapipe_detect_single(frame, landmarkers, frame_num)
        draw_landmarks(frame, results)
        keypoints = mediapipe_extract_single(results)
        keypoints = keypoints.flatten()
        keypoints = apply_weighting_to_flattened(keypoints)
        res = model.predict([np.array([keypoints])])[0]
        pred_class = np.argmax(res)
        pred_word = dictionary[pred_class]
        predictions.append(pred_word)
        if len(predictions) > 20:
            predictions = predictions[-20:]
        
        if len(sentence) == 0:
            sentence.append(pred_class)
        elif pred_class != sentence[-1]:
            sentence.append(pred_class)

        for i, probabilty in enumerate(res):
            cv2.putText(frame, f"{dictionary[i]}: {probabilty:.2f}", (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA) 
        # cv2.rectangle(frame, (10, 10), (200, 10 + 30), (0, 255, 0), -1)
        # cv2.putText(frame, f"Prediction: {pred_word}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)
        # cv2.rectangle(frame, (10, 40), (200, 40 + 30), (0, 255, 0), -1)
        # cv2.putText(frame, f"Sentence: {' '.join([str(x) for x in sentence])}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


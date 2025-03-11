import cv2
import os
import numpy as np

from config import config_parser
from mp_util import init_landmarkers, mediapipe_detect_single, mediapipe_extract_single, draw_landmarks

if __name__ == "__main__":
    config = config_parser("config/config_image.yaml")

    capture_source = config.capture.source
    cap = cv2.VideoCapture(capture_source)

    width = int(config.capture.resolution_width)
    height = int(config.capture.resolution_height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    labels = config.dictionary
    landmarkers = init_landmarkers()

    cur_label_index = 0
    cur_label_count = 0

    if cap.isOpened():
        frame_count = 0
        while True:
            frame_count += 1
            ret, frame = cap.read()
            if not ret:
                break
            mediapipe_results = mediapipe_detect_single(frame, landmarkers, frame_count)
            frame_copy = frame.copy() 
            draw_landmarks(frame_copy, mediapipe_results)
            
            if frame_count % 30 == 0:
                keypoints = mediapipe_extract_single(mediapipe_results)
                keypoint_path = os.path.join(config.paths.keypoints, labels[cur_label_index])
                if not os.path.exists(keypoint_path):
                    os.makedirs(keypoint_path)
                np.save(f"{keypoint_path}/{cur_label_count}.npy", keypoints)

                image_path = os.path.join(config.paths.dataset, labels[cur_label_index])
                if not os.path.exists(image_path):
                    os.makedirs(image_path)
                cv2.imwrite(f"{image_path}/{cur_label_count}.jpg", frame)

                visualization_path = os.path.join(r"F:\dataset\minified_2\visualization", labels[cur_label_index])
                if not os.path.exists(visualization_path):
                    os.makedirs(visualization_path)
                cv2.imwrite(f"{visualization_path}/{cur_label_count}.jpg", frame_copy)

                cur_label_count += 1
                if cur_label_count == 30:
                    cur_label_index += 1
                    cur_label_count = 0

                if cur_label_index == len(labels):
                    break

            cv2.putText(frame_copy, "Label: {}, Count: {}/30".format(labels[cur_label_index], cur_label_count+1), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
            cv2.putText(frame_copy, "Next label in: {}".format(30 - cur_label_count), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
            cv2.putText(frame_copy, "Next label: {}".format(labels[cur_label_index+1] if cur_label_index+1 < len(labels) else ""), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
            cv2.imshow("Camera Feed", frame_copy)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
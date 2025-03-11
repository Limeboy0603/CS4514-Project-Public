import cv2
import numpy as np
import os

from config import config_parser
from mp_util import init_landmarkers, mediapipe_detect_single, mediapipe_extract_single, draw_landmarks

def main():
    config = config_parser("config/config_clip.yaml")

    capture_source = config.capture.source
    cap = cv2.VideoCapture(capture_source)

    # camera settings
    width = int(config.capture.resolution_width)
    height = int(config.capture.resolution_height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    labels = config.dictionary
    landmarkers = init_landmarkers()

    if cap.isOpened():
        frame_count = 0
        total_frames = 0
        for label in labels:
            os.makedirs(os.path.join(config.paths.keypoints, label), exist_ok=True)
            os.makedirs(os.path.join(config.paths.dataset, label), exist_ok=True)
            for sequence in range(-1, config.sequence.count):
                keypoints = []
                frames = []
                keypoint_path = os.path.join(config.paths.keypoints, label, f"{sequence}.npy")
                video_path = os.path.join(config.paths.dataset, label, f"{sequence}.avi")
                out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
                for frame_count in range(config.sequence.frame):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    total_frames += 1
                    frames.append(frame)
                    out.write(frame)
                    mediapipe_results = mediapipe_detect_single(frame, landmarkers, total_frames)
                    draw_landmarks(frame, mediapipe_results)
                    keypoint = mediapipe_extract_single(mediapipe_results)
                    keypoints.append(keypoint)
                    # if frame_count == 0:
                    #     cv2.putText(frame, "STARTING COLLECTION", (width/2, height/2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, f"Label: {label}, Count: {sequence+1 if sequence!= -1 else 'PREP'}/{config.sequence.count}, Frame: {frame_count+1}/{config.sequence.frame}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.imshow("frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        exit()
                if sequence != -1:
                    out.release()
                    np.save(keypoint_path, np.array(keypoints))
                    print(f"Saved {keypoint_path}")
                    print(f"Saved {video_path}")

                # add wait time
                cv2.waitKey(500)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
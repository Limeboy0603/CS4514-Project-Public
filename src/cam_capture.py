import cv2
import os

from config import config_parser

# removes all continuous keypoint extraction functions and only save the video
def main(config_path: str, demo: bool = False):
    config = config_parser(config_path)

    capture_source = config.capture.source
    cap = cv2.VideoCapture(capture_source)

    # camera settings
    width = int(config.capture.resolution_width)
    height = int(config.capture.resolution_height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    labels = config.dictionary

    if cap.isOpened():
        frame_count = 0
        for label in labels:
            os.makedirs(os.path.join(config.paths.dataset, label), exist_ok=True)
            for sequence in range(70, config.sequence.count):
                frames = []
                video_path = os.path.join(config.paths.dataset, label, f"{sequence}.avi")
                if not demo:
                    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
                for frame_count in range(config.sequence.frame):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                    if not demo:
                        out.write(frame)
                    cv2.putText(frame, f"Label: {label}, Count: {sequence+1 if sequence!= -1 else 'PREP'}/{config.sequence.count}, Frame: {frame_count+1}/{config.sequence.frame}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.imshow("frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        exit()
                if sequence != -1:
                    if not demo:
                        out.release()
                cv2.waitKey(500)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main("config/config_clip.yaml" if os.name == "nt" else "config/config_clip_linux.yaml", False)
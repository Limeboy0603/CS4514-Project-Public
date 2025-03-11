import cv2
from mediapipe.python.solutions import drawing_utils
from mediapipe.python.solutions.holistic import Holistic
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS
import pafy

def get_video_youtube(url):
    video = pafy.new(url)
    best = video.getbest(preftype="mp4")
    return best.url

if __name__ == "__main__":
    # 0 for camera
    # 1 for OBS virtual camera
    # or a youtube video link
    # cap = cv2.VideoCapture(1)
    cap = cv2.VideoCapture(get_video_youtube("https://www.youtube.com/watch?v=U52l-VLQ7AE"))

    # camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    with Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            _, frame = cap.read()

            # check if frame is empty
            if not _:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                drawing_utils.draw_landmarks(
                    frame, results.face_landmarks, FACEMESH_TESSELATION
                )
                drawing_utils.draw_landmarks(
                    frame, results.left_hand_landmarks, HAND_CONNECTIONS
                )
                drawing_utils.draw_landmarks(
                    frame, results.right_hand_landmarks, HAND_CONNECTIONS
                )
                drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, POSE_CONNECTIONS
                )

            cv2.imshow('Camera Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # destroy the camera and all windows
    cap.release()
    cv2.destroyAllWindows()
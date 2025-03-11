import cv2
from mp_util import init_landmarkers, mediapipe_detect_single, mediapipe_extract_single, draw_landmarks

if __name__ == "__main__":
    landmarkers = init_landmarkers()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    frame_count = 1
    assert cap.isOpened(), "Error: Camera not opened."
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = mediapipe_detect_single(frame, landmarkers, frame_count)
        keypoints = mediapipe_extract_single(results)
        print(keypoints)
        draw_landmarks(frame, results)
        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()
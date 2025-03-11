import cv2
from mp_util import init_landmarkers, mediapipe_detect_single, mediapipe_extract_single, draw_landmarks

if __name__ == "__main__":
    landmarkers = init_landmarkers()
    cap = cv2.VideoCapture(2)
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

    # frame = cv2.imread(r"F:\dataset\minified_2\visualization\BLANK\0.jpg")
    # landmarkers = init_landmarkers()
    # frame_counter = 1
    # while True:
    #     results = mediapipe_detect_single(frame, landmarkers, frame_counter)
    #     keypoints = mediapipe_extract_single(results)
    #     print(keypoints)
    #     draw_landmarks(frame, results)
    #     cv2.imshow("Camera Feed", frame)
    #     frame_counter += 1
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()


# get the FPS of the camera
# import cv2
# import time

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# frame_count = 0
# start_time = time.time()
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     cv2.imshow("Camera Feed", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     frame_count += 1
# end_time = time.time()
# fps = frame_count / (end_time - start_time)
# print(f"FPS: {fps}")
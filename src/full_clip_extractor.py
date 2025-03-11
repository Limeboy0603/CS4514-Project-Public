import cv2
import os

video_path = "dataset/minified_3_fixed/full_test/final.avi"
frame_save_path = "dataset/minified_3_fixed/full_test/frames"
if os.name == "nt":
    video_path = video_path.replace("/", "\\")
    frame_save_path = frame_save_path.replace("/", "\\")

cap = cv2.VideoCapture(video_path)
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(os.path.join(frame_save_path, f"{frame_count}.png"), frame)
    frame_count += 1
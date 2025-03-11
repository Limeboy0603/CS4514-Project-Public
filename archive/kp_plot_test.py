# simple testing script to see whether keypoints match the actual location of body parts in the frame

import numpy
import cv2

from .cv_util import preprocess_image, video_to_frame_list
from mp_util_legacy import preprocess_keypoints_multiple, get_raw_coords, mediapipe_extract_multiple, create_holistic

video_file = r"dataset\minified_3_fixed\clips\English\60.avi"
frames = video_to_frame_list(video_file)

# holistic = create_holistic()
# keypoints = mediapipe_extract_multiple(frames, visualize=False)
keypoints = numpy.load(r"dataset\minified_3_fixed\keypoints\English\60.npy")

# angle = 15
# angle = 0
# tx = numpy.random.uniform(-0.2, 0.2)
# ty = numpy.random.uniform(-0.2, 0.2)
# scale = numpy.random.uniform(0.8, 1.2)

# tx_image = tx * 1920
# ty_image = ty * 1080

# transformed_video = []
# for frame in frames:
#     transformed_frame = preprocess_image(frame, angle=angle, tx=tx_image, ty=ty_image, scale=(scale, scale))
#     transformed_video.append(transformed_frame)
# print(len(transformed_video))

# convert angle from degrees to radians
# angle = numpy.radians(angle)

# transformed_keypoints = preprocess_keypoints_multiple(keypoints, angle=angle, tx=tx, ty=ty, scale=scale)

# plot the first frame and first frame's keypoints
pose_keypoints, face_keypoints, left_hand_keypoints, right_hand_keypoints = get_raw_coords(keypoints[0])
# pose_keypoints, face_keypoints, left_hand_keypoints, right_hand_keypoints = get_raw_coords(transformed_keypoints[0])

frame = frames[0]
# frame = transformed_video[0]
# x and y are coordinates normalized to [0, 1]
for x, y in pose_keypoints:
    cv2.circle(frame, (int(x * 1920), int(y * 1080)), 3, (0, 255, 0), -1)
for x, y, _ in face_keypoints:
    cv2.circle(frame, (int(x * 1920), int(y * 1080)), 3, (255, 0, 0), -1)
for x, y, _ in left_hand_keypoints:
    cv2.circle(frame, (int(x * 1920), int(y * 1080)), 3, (0, 0, 255), -1)
for x, y, _ in right_hand_keypoints:
    cv2.circle(frame, (int(x * 1920), int(y * 1080)), 3, (0, 0, 255), -1)

cv2.imshow("frame", frame)
cv2.waitKey(0)
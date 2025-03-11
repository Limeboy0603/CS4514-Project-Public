import cv2
import os

dataset_clip_path = "dataset/minified_3_fixed/clips"
if os.name == "nt":
    dataset_clip_path = dataset_clip_path.replace("/", "\\")
dataset_classes = [
    "Hello",
    "Me",
    "English",
    "Name",
    "D",
    "I",
    "C",
    "K"
]

frames = []

for dataset_class in dataset_classes:
    clip_path = os.path.join(dataset_clip_path, dataset_class, "0.avi")
    print(clip_path)
    cap = cv2.VideoCapture(clip_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

print(len(frames))

# export frames to a video
save_path = "dataset/minified_3_fixed/full_test/final.avi"
if os.name == "nt":
    save_path = save_path.replace("/", "\\")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (frames[0].shape[1], frames[0].shape[0]))
for frame in frames:
    out.write(frame)

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import ruptures as rpt
import pandas as pd
import cv2

# set environment variable
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

def get_transition_points(keypoints, n_gestures):
    # Normalize the keypoints data
    scaler = StandardScaler()
    keypoints_normalized = scaler.fit_transform(keypoints)

    # Apply K-Means clustering
    model = rpt.Binseg(model="rbf").fit(keypoints_normalized)
    # model = rpt.Binseg(model="rbf").fit(keypoints)
    breakpoints = model.predict(n_bkps=n_gestures - 1)  # Number of breakpoints is n_segments - 1

    del model
    del scaler

    return breakpoints

# keypoint_dir = r"F:\dataset\tvb-hksl-news\keypoints_mediapipe"
# frame_dir = r"F:\dataset\tvb-hksl-news\frames"
# cluster_dir = r"F:\dataset\tvb-hksl-news\clusters"
keypoint_dir = "./dataset/tvb-hksl-news/keypoints_mediapipe"
frame_dir = "./dataset/tvb-hksl-news/frames"
cluster_dir = "./dataset/tvb-hksl-news/clusters/binseg"

os.makedirs(cluster_dir, exist_ok=True)

# split_file_list = os.listdir(r"F:\dataset\tvb-hksl-news\split")
split_file_list = os.listdir("./dataset/tvb-hksl-news/split")

forucc = cv2.VideoWriter_fourcc(*'mp4v')

for split_file in split_file_list:
    # split files are CSV files with | as delimiter
    # split_data = pd.read_csv(os.path.join(r"F:\dataset\tvb-hksl-news\split", split_file), delimiter="|")
    split_data = pd.read_csv(os.path.join("./dataset/tvb-hksl-news/split", split_file), delimiter="|")
    
    for index, row in split_data.iterrows():
        sample_id = row["id"]
        sample_gloss_length = len(row["glosses"].split())
        sample_gloss = row["glosses"].split()

        # if there are repeated glosses, replace them with gloss_1, gloss_2, etc.
        gloss_count = {}
        for i, gloss in enumerate(sample_gloss):
            if gloss in gloss_count:
                gloss_count[gloss] += 1
                sample_gloss[i] = f"{gloss}_{gloss_count[gloss]}"
            else:
                gloss_count[gloss] = 1

        date = sample_id.split("/")[0]
        name = sample_id.split("/")[1]
        os.makedirs(os.path.join(cluster_dir, date, name), exist_ok=True)
        vid_save_dir = os.path.join(cluster_dir, date, name)

        # sample_id = sample_id.replace("/", "\\")
        keypoints = np.load(os.path.join(keypoint_dir, sample_id + ".npy"))
        transitions = get_transition_points(keypoints, sample_gloss_length)

        # for each transition segments, save the respective frames to a video
        frames_list = os.listdir(os.path.join(frame_dir, sample_id))
        frames_list.sort(key=lambda x: int(os.path.splitext(x)[0]))
        start_frame = 0
        for i, transition in enumerate(transitions):
            end_frame = transition
            frames = []
            for frame_index in range(start_frame, end_frame):
                frame = cv2.imread(os.path.join(frame_dir, sample_id, frames_list[frame_index]))
                frames.append(frame)
            out = cv2.VideoWriter(os.path.join(vid_save_dir, f"{sample_gloss[i]}.mp4"), forucc, 30, (frame.shape[1], frame.shape[0]))
            for frame in frames:
                out.write(frame)
            out.release()
            start_frame = end_frame
        
        print(f"Processed: {sample_id}")
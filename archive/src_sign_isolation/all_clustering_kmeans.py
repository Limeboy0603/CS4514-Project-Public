import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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
    kmeans = KMeans(n_clusters=n_gestures, random_state=42)
    clusters = kmeans.fit_predict(keypoints_normalized)
    
    # Clean up
    del scaler
    del keypoints_normalized
    del kmeans

    # Identify transition points
    transitions = np.where(np.diff(clusters) != 0)[0] + 1

    # Ensure exactly n_gestures - 1 transition
    if len(transitions) > n_gestures - 1:
        # Calculate the magnitude of changes at each transition
        changes = np.abs(np.diff(clusters[transitions - 1]))
        # Select the top n_gestures - 1 transitions with the largest changes
        top_transitions_indices = np.argsort(changes)[- (n_gestures - 1):]
        transitions = transitions[top_transitions_indices]
        transitions.sort()

    return transitions

keypoint_dir = r"F:\dataset\tvb-hksl-news\keypoints_mediapipe"
frame_dir = r"F:\dataset\tvb-hksl-news\frames"
cluster_dir = r"F:\dataset\tvb-hksl-news\clusters"

os.makedirs(cluster_dir, exist_ok=True)

split_file_list = os.listdir(r"F:\dataset\tvb-hksl-news\split")

forucc = cv2.VideoWriter_fourcc(*'mp4v')

for split_file in split_file_list:
    # split files are CSV files with | as delimiter
    split_data = pd.read_csv(os.path.join(r"F:\dataset\tvb-hksl-news\split", split_file), delimiter="|")
    
    for index, row in split_data.iterrows():
        sample_id = row["id"]
        sample_gloss = row["glosses"].split()
        sample_gloss = list(set(sample_gloss))
        sample_gloss_length = len(sample_gloss)

        date = sample_id.split("/")[0]
        name = sample_id.split("/")[1]
        os.makedirs(os.path.join(cluster_dir, date, name), exist_ok=True)
        vid_save_dir = os.path.join(cluster_dir, date, name)

        sample_id = sample_id.replace("/", "\\")
        keypoints = np.load(os.path.join(keypoint_dir, sample_id + ".npy"))
        transitions = get_transition_points(keypoints, sample_gloss_length)

        # for each transition segments, save the respective frames to a video
        frames_list = os.listdir(os.path.join(frame_dir, sample_id))
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
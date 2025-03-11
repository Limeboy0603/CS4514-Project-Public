import os
import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd
import cv2

# set environment variable
# os.environ["LOKY_MAX_CPU_COUNT"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"

def get_cluster_means(keypoints, n_gestures):
    # Normalize the keypoints data
    # scaler = StandardScaler()
    # keypoints_normalized = scaler.fit_transform(keypoints)

    # Apply K-Means clustering
    gmm = GaussianMixture(n_components=n_gestures, random_state=42)
    clusters = gmm.fit_predict(keypoints)

    # transitions = np.where(np.diff(clusters) != 0)[0] + 1
    # segment_ids = clusters[transitions]
    # segment_ids = np.concatenate([[clusters[0]], segment_ids])

    # get the keypoint closest to the mean of each cluster
    # and return the index of that keypoint for each cluster
    cluster_means = gmm.means_
    closest_keypoints = []
    for cluster_mean in cluster_means:
        closest_keypoint = np.argmin(np.linalg.norm(keypoints - cluster_mean, axis=1))
        closest_keypoints.append(closest_keypoint)

    return closest_keypoints


# keypoint_dir = r"F:\dataset\tvb-hksl-news\keypoints_mediapipe"
# frame_dir = r"F:\dataset\tvb-hksl-news\frames"
# cluster_dir = r"F:\dataset\tvb-hksl-news\clusters"
keypoint_dir = "./dataset/tvb-hksl-news/keypoints_mediapipe"
frame_dir = "./dataset/tvb-hksl-news/frames"
cluster_dir = "./dataset/tvb-hksl-news/clusters/gmm"

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
        sample_gloss = row["glosses"].split()
        uniq_sample_gloss = list(set(sample_gloss))
        sample_gloss_length = len(uniq_sample_gloss)

        # if there are repeated glosses, replace them with gloss_1, gloss_2, etc.
        # gloss_count = {}
        # for i, gloss in enumerate(sample_gloss):
        #     if gloss in gloss_count:
        #         gloss_count[gloss] += 1
        #         sample_gloss[i] = f"{gloss}_{gloss_count[gloss]}"
        #     else:
        #         gloss_count[gloss] = 1

        date = sample_id.split("/")[0]
        name = sample_id.split("/")[1]
        os.makedirs(os.path.join(cluster_dir, date, name), exist_ok=True)
        mean_frame_save_dir = os.path.join(cluster_dir, date, name)

        # sample_id = sample_id.replace("/", "\\")
        keypoints = np.load(os.path.join(keypoint_dir, sample_id + ".npy"))
        cluster_means = get_cluster_means(keypoints, sample_gloss_length)

        # for each cluster mean, extract the respective frame
        cur_frame_dir = os.path.join(frame_dir, sample_id)
        frames_list = os.listdir(cur_frame_dir)
        frames_list.sort(key=lambda x: int(os.path.splitext(x)[0]))
        for i, cluster_mean in enumerate(cluster_means):
            frame = cv2.imread(os.path.join(cur_frame_dir, frames_list[cluster_mean]))
            cv2.imwrite(os.path.join(mean_frame_save_dir, f"{uniq_sample_gloss[i]}.jpg"), frame)

        print(f"Processed: {sample_id}")
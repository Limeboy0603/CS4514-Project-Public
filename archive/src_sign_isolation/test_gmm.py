import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your keypoints data
keypoints = np.load(r"F:\dataset\tvb-hksl-news\keypoints_mediapipe\2020-07-05\024559-024662.npy")  # Shape: (timestamp, number of features)

# Normalize the keypoints data
scaler = StandardScaler()
keypoints_normalized = scaler.fit_transform(keypoints)

# Apply K-Means clustering
n_gestures = 13  # Number of gestures (clusters)
# kmeans = KMeans(n_clusters=n_gestures, random_state=42)
# clusters = kmeans.fit_predict(keypoints_normalized)
gmm = GaussianMixture(n_components=n_gestures, random_state=42)
clusters = gmm.fit_predict(keypoints_normalized)

# Identify transition points
transitions = np.where(np.diff(clusters) != 0)[0] + 1
# get cluster ids for each segment
segment_ids = clusters[transitions]
# also includes the segment of the first segment
segment_ids = np.concatenate([[clusters[0]], segment_ids])

# Print the transition points
print("Transition points:", transitions)
print("Segment IDs:", segment_ids)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(keypoints_normalized)
for transition in transitions:
    plt.axvline(x=transition, color='r', linestyle='--')
plt.title('Keypoints with Transition Points')
plt.xlabel('Timestep')
plt.ylabel('Normalized Keypoints')
plt.show()
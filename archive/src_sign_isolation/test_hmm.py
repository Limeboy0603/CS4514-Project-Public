import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from hmmlearn import hmm

# Load your keypoints data
keypoints = np.load(r"F:\dataset\tvb-hksl-news\keypoints_mediapipe\2020-07-05\024559-024662.npy")  # Shape: (timestamp, number of features)

# Normalize the keypoints data
scaler = StandardScaler()
keypoints_normalized = scaler.fit_transform(keypoints)

# Define the number of states (gestures)
n_gestures = 13

# Train HMM
model = hmm.GaussianHMM(n_components=n_gestures, covariance_type="diag", n_iter=100, random_state=42)
model.fit(keypoints_normalized)

# Predict the hidden states (gestures)
hidden_states = model.predict(keypoints_normalized)

# Identify transition points
transitions = np.where(np.diff(hidden_states) != 0)[0] + 1
# get cluster ids for each segment
segment_ids = hidden_states[transitions]
# also includes the segment of the first segment
segment_ids = np.concatenate([[hidden_states[0]], segment_ids])

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
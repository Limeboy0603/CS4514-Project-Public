import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import ruptures as rpt

# Load your keypoints data
keypoints = np.load(r"F:\dataset\tvb-hksl-news\keypoints_mediapipe\2020-07-21\017386-017665.npy")  # Shape: (timestamp, number of features)

# Normalize the keypoints data
scaler = StandardScaler()
keypoints_normalized = scaler.fit_transform(keypoints)

# Define the number of segments
n_segments = 26

# Apply Window-Based algorithm for change point detection
model_window = rpt.Window(width=5, model="rbf").fit(keypoints_normalized)
breakpoints_window = model_window.predict(n_bkps=n_segments - 1)  # Number of breakpoints is n_segments - 1

# Print the breakpoints
print(len(breakpoints_window))
print("Breakpoints (Window-Based):", breakpoints_window)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(keypoints_normalized)
for breakpoint in breakpoints_window:
    plt.axvline(x=breakpoint, color='r', linestyle='--')
plt.title('Keypoints with Breakpoints (Window-Based)')
plt.show()
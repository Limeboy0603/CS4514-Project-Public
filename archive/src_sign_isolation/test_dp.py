import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import ruptures as rpt

# Load your keypoints data
keypoints = np.load(r"F:\dataset\tvb-hksl-news\keypoints_mediapipe\2020-07-21\017386-017665.npy")
# 2020-07-21/017386-017665
# D S E 文 考試 明天 說 這 四 十 百分比 入 香港 本地 全部 香港 大 七 <BAD> 考試 五 部分 五 <BAD> 星 星
# 文憑試明日放榜 超過四成日校考生 考獲入讀本地大學的成績 七人考到七科5
# D+S+E+文+考試(=香港中學文憑考試) 明天 公佈 這 四十+百分比(=四成) 入 香港 本地 全部 香港 大 七 BAD-SEGMENT 考試 五 部份 五 BAD-SEGMENT 星 星

# Normalize the keypoints data
scaler = StandardScaler()
keypoints_normalized = scaler.fit_transform(keypoints)

# Define the number of segments
n_segments = 26

# Apply Dynamic Programming algorithm for change point detection
model_dp = rpt.Dynp(model="rbf", min_size=2, jump=1).fit(keypoints_normalized)
breakpoints_dp = model_dp.predict(n_bkps=n_segments - 1)  # Number of breakpoints is n_segments - 1

# Print the breakpoints
print("Breakpoints (DP):", breakpoints_dp)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(keypoints_normalized)
for breakpoint in breakpoints_dp:
    plt.axvline(x=breakpoint, color='r', linestyle='--')
plt.title('Keypoints with Breakpoints (DP)')
plt.xlabel('Timestep')
plt.ylabel('Normalized Keypoints')
plt.show()
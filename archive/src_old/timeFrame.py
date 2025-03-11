'''
This python program is used to extract frames (per second) from a video file.

What it does:
- Process all videos in a folder or process a single video file by specifying its name.
- Create individual folders for each video to store the extracted frames.
- Extract frames at 1-second intervals and save them as JPG images.

Note:
- install the moviepy library using "pip install moviepy"
- open a folder named "Unprocessed" and put the video files (.MP4) in it.
'''

import os
import tqdm
from moviepy.editor import VideoFileClip
import imageio

def extract_frames(video_path, output_dir, fps=1):
  """
  Extracts frames from a video at a specified frame rate.

  Args:
      video_path (str): Path to the video file.
      output_dir (str): Path to the directory where the extracted frames will be saved.
      fps (int, optional): The number of frames to extract per second. Defaults to 1.

  Returns:
      None
  """

  video = VideoFileClip(video_path)
  num_frames = int(video.duration * fps)

  # Create output directory if it doesn't exist
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  with tqdm.tqdm(total=num_frames, desc=f"Processing {os.path.basename(video_path)}") as pbar:
    for i in range(num_frames):
      # Extract frame efficiently using get_frame()
      frame = video.get_frame(i / fps)

      # Save frame with appropriate filename
      frame_filename = os.path.join(output_dir, f"frame_{i:05d}.jpg")
      imageio.imwrite(frame_filename, frame)

      pbar.update()

  video.close()

if __name__ == "__main__":
  # Assuming the videos are in a folder named "Unprocessed"
  input_folder = "Unprocessed"
  for filename in os.listdir(input_folder):
    if filename.endswith(".mp4"):
      video_path = os.path.join(input_folder, filename)
      output_dir = os.path.join(input_folder, os.path.splitext(filename)[0])
      extract_frames(video_path, output_dir)
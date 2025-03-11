# perform train val test split

import os
import random
import shutil
from config import config_parser

def split_dataset(dataset_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Ensure the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"

    # Create output directories if they don't exist
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

    # List all class directories
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

    for class_name in classes:
        class_dir = os.path.join(dataset_dir, class_name)
        videos = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]

        # Shuffle the list of videos
        random.shuffle(videos)

        # Calculate split indices
        train_end = int(train_ratio * len(videos))
        val_end = train_end + int(val_ratio * len(videos))

        # Split the videos
        train_videos = videos[:train_end]
        val_videos = videos[train_end:val_end]
        test_videos = videos[val_end:]

        # Move or copy the files to the respective directories
        for split, split_videos in zip(['train', 'val', 'test'], [train_videos, val_videos, test_videos]):
            split_class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            for video in split_videos:
                src = os.path.join(class_dir, video)
                dst = os.path.join(split_class_dir, video)
                shutil.copy(src, dst)  # Use shutil.move(src, dst) if you want to move instead of copy

def main(config_path: str, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # config = config_parser("config/config_clip.yaml")
    config = config_parser(config_path)
    dataset_dir = config.paths.keypoints
    output_dir = config.paths.split
    # clear the output directory
    shutil.rmtree(output_dir, ignore_errors=True)
    split_dataset(dataset_dir, output_dir, train_ratio, val_ratio, test_ratio)

if __name__ == "__main__":
    main("config/config_clip.yaml" if os.name == "nt" else "config/config_clip_linux.yaml")
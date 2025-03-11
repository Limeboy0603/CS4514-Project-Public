# Performs a full run of all scripts in this project
# Make sure you have set everything in the config file correctly and have all libraries installed

import os
os.environ['KERAS_BACKEND'] = 'torch'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from argparse import ArgumentParser

import cam_capture
import feat_extraction
import dataset_split
import model_train
import model_test
import model_use_isolated
import llm_train

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config_clip.yaml" if os.name == "nt" else "config/config_clip_linux.yaml", help="Path to config file")
    parser.add_argument("--capture", action="store_true", help="Capture video and save to dataset. If set to false, feature extraction and training scripts will use whatever is in the dataset folder.")
    parser.add_argument("--extractsplit", action="store_true", help="Perform feature extraction and dataset splitting.")
    parser.add_argument("--use", action="store_true", help="Use the model right after training")
    args = parser.parse_args()

    # Alert the user if config is not provided
    if args.config is None:
        print("Config file path not found, defaulting to config/config_clip.yaml")
    else:
        print(f"Using config file: {args.config}")

    # Throw if config file does not exist
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at {args.config}")

    # Capturing a dataset of clips, class names are defined in the config file
    if args.capture:
        print("Starting dataset capture")
        cam_capture.main(args.config)

    if args.extractsplit:
        # Extracting keypoints from the captured clips
        feat_extraction.main(args.config)
        
        # Splitting the dataset into train, validation, and test sets
        # Optional parameters:
        # train_ratio: float = 0.8 | Ratio of the dataset to be used for training
        # val_ratio: float = 0.1 | Ratio of the dataset to be used for validation
        # test_ratio: float = 0.1 | Ratio of the dataset to be used for testing
        dataset_split.main(args.config)

    # Training the model
    model_train.main(args.config)

    # Performing predictions on the testing set
    # Will print out the prediction of each clip and the accuracy of the model
    model_test.main(args.config)

    # Building the bart model
    llm_train.main()

    # Using the model
    if args.use:
        model_use_isolated.main(args.config)
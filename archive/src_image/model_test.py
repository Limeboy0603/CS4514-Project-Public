import os
os.environ["GLOG_minloglevel"] ="3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import keras
import numpy as np
from config import config_parser
import cv2
from archive.cv_util import preprocess_image
from mp_util import init_landmarkers, mediapipe_detect_single, mediapipe_extract_single, draw_landmarks, preprocess_keypoints, apply_weighting_to_flattened

class ValidationGenerator(keras.utils.Sequence):
    # X: keypoint files named 0.npy to 29.npy in each directory
    # Y: name of each directory
    def __init__(self, keypoint_path, batch_size, randomize=False):
        self.keypoint_path = keypoint_path
        self.batch_size = batch_size
        self.randomize = randomize
        
        self.file_list = []
        self.labels = []

        self.all_labels = sorted(os.listdir(keypoint_path))
        for label in self.all_labels:
            for i in range(30):
                self.file_list.append(os.path.join(label, f"{i}.npy"))
                self.labels.append(label)

        self.indices = range(len(self.file_list))
        self.on_epoch_end()

    def on_epoch_end(self):
        with open("debug.txt", "a") as f:
            f.write("on_epoch_end has been called\n")
        self.indices = np.random.permutation(self.indices)

    def __len__(self):
        return int(np.floor(len(self.file_list) / self.batch_size))
    
    def __data_generation(self, indices):
        labels = []
        keypoints = []

        for index in indices:
            keypoint_file_name = self.file_list[index]
            label = self.labels[index]
            keypoint_file_name = os.path.join(self.keypoint_path, keypoint_file_name)
            # write debug message to debug.txt
            with open("debug.txt", "a") as f:
                f.write(f"{label}: {keypoint_file_name}\n")
            kp = np.load(keypoint_file_name, mmap_mode="r")
            if self.randomize:
                angle = np.random.randint(-10, 10)
                tx = np.random.uniform(-0.3, 0.3)
                ty = np.random.uniform(-0.1, 0.1)
                tz = np.random.uniform(-0.1, 0.1)
                scale = np.random.uniform(0.8, 1.2)
                kp = preprocess_keypoints(kp, angle, tx, ty, tz, scale)
            kp = kp.reshape(-1)
            kp = apply_weighting_to_flattened(kp)
            keypoints.append(kp)
            labels.append(label)

        X = np.array(keypoints)
        Y = np.array([self.all_labels.index(label) for label in labels])
        # if self.randomize: print("---> Batch generated")
        return X, Y
    
    def __getitem__(self, index):
        inds = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X, Y = self.__data_generation(inds)
        return X, Y
    
if __name__ == "__main__":
    config = config_parser("config/config_image.yaml")
    validation_generator = ValidationGenerator(config.paths.keypoints, 32, False)

    model = keras.models.load_model(config.paths.model)

    pred_dict = {}
    for label in config.dictionary:
        pred_dict[label] = []

    for i in range(len(validation_generator)):
        X, Y = validation_generator[i]

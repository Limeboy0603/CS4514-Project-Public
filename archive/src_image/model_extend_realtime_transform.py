import tensorflow as tf
import keras
import os
import numpy as np
from config import config_parser
import cv2
from archive.cv_util import preprocess_image
from mp_util import init_landmarkers, mediapipe_detect_single, mediapipe_extract_single, draw_landmarks

class DataGenerator(keras.utils.Sequence):
    def __init__(self, image_path, batch_size, shuffle=True):
        self.image_path = image_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.file_list = []
        self.labels = []

        self.all_labels = sorted(os.listdir(image_path))
        for label in self.all_labels:
            for file in os.listdir(os.path.join(image_path, label)):
                self.file_list.append(file)
                self.labels.append(label)
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.file_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.file_list) / self.batch_size))
    
    def __data_generation(self, indices):
        labels = []
        images = []
        keypoints = []

        for index in indices:
            image_file_name = self.file_list[index]
            label = self.labels[index]
            images.append(cv2.imread(os.path.join(self.image_path, label, image_file_name)))
            labels.append(label)

        for i in range(len(images)):
            while True:
                angle = np.random.randint(-15, 15)
                tx = np.random.randint(-10, 10)
                ty = np.random.randint(-10, 10)
                scale_x = np.random.uniform(0.8, 1.2)
                scale_y = np.random.uniform(0.8, 1.2)
                transformed_image = preprocess_image(images[i], angle=angle, tx=tx, ty=ty, scale=(scale_x, scale_y))
                landmarkers = init_landmarkers()
                mp_results = mediapipe_detect_single(transformed_image, landmarkers, 0)
                keypoint = mediapipe_extract_single(mp_results)
                # brute-force approach
                # if all face-related keypoints (index: range(33, 125)) are 0, then the face is not detected
                if np.all(keypoint[33:125] == 0):
                    del landmarkers
                    print(f"No face detected, retrying label {label} id {index} ...")
                    continue
                del landmarkers
                keypoints.append(keypoint)
                break

        for i in range(len(keypoints)):
            keypoints[i] = keypoints[i].reshape(-1)

        X = np.array(keypoints)
        Y = np.array([self.all_labels.index(label) for label in labels])
        return X, Y

    def __getitem__(self, index):
        indices = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X, Y = self.__data_generation(indices)
        return X, Y

if __name__ == "__main__":
    config = config_parser("config/config_image.yaml")
    
    training_generator = DataGenerator(config.paths.dataset, 32, True)

    # checkpoints
    checkpoint = keras.callbacks.ModelCheckpoint(config.paths.model_checkpoint, monitor="val_loss", save_best_only=True)

    # callbacks
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

    # load model and extend training
    model = keras.models.load_model(config.paths.model)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(training_generator, epochs=256, initial_epoch=128, callbacks=[checkpoint, early_stopping])
    model.save(config.paths.model)
    print(model.history.history)

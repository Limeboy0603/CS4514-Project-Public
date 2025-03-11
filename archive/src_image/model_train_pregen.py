import tensorflow as tf
import keras
import os
import numpy as np
from config import config_parser

class DataGenerator(keras.utils.Sequence):
    def __init__(self, keypoint_path, batch_size, shuffle=True):
        self.keypoint_path = keypoint_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.file_list = []
        self.labels = []

        self.all_labels = sorted(os.listdir(keypoint_path))
        for label in self.all_labels:
            for file in os.listdir(os.path.join(keypoint_path, label)):
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
        image_features = []

        for index in indices:
            kp_file_name = self.file_list[index]
            label = self.labels[index]
            image_features.append(np.load(os.path.join(self.keypoint_path, label, kp_file_name), mmap_mode="r"))
            labels.append(label)

        for i in range(len(image_features)):
            image_features[i] = image_features[i].reshape(-1)

        X = np.array(image_features)
        Y = np.array([self.all_labels.index(label) for label in labels])
        return X, Y
    
    def __getitem__(self, index):
        indices = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X, Y = self.__data_generation(indices)
        return X, Y

if __name__ == "__main__":
    config = config_parser("config/config_image.yaml")
    
    training_generator = DataGenerator(config.paths.keypoints, 32, True)

    model = keras.models.Sequential([
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(len(training_generator.all_labels), activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(training_generator, epochs=16)
    model.save(config.paths.model)
    print(model.history.history)
    model.summary()
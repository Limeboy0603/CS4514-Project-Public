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

class DataGenerator(keras.utils.Sequence):
    def __init__(self, image_path, label_list, batch_size, shuffle=True):
        self.image_path = image_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.file_list = []
        self.labels = []

        for label in self.label_list:
            for file in os.listdir(os.path.join(image_path, label)):
                self.file_list.append(file)
                self.labels.append(label)

        self.indexes = np.arange(len(self.file_list))

    def on_epoch_end(self):
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
                reflection = np.random.choice([True, False])
                transformed_image = preprocess_image(images[i], angle=angle, tx=tx, ty=ty, scale=(scale_x, scale_y), reflection=reflection)
                landmarkers = init_landmarkers()
                mp_results = mediapipe_detect_single(transformed_image, landmarkers, 0)
                keypoint = mediapipe_extract_single(mp_results)
                # brute-force approach
                # if all face-related keypoints (index: range(33, 125)) are 0, then the face is not detected
                if np.all(keypoint[33:125] == 0):
                    del landmarkers
                    print(f"No face detected, retrying label {label} id {index} ...")
                    continue
                # only allows the hand to not be detected if the label is BLANK
                # hand covers the last 42*3 keypoints
                if labels[i] != "BLANK" and np.all(keypoint[125:] == 0):
                    del landmarkers
                    print(f"No hand detected, retrying label {label} id {index} ...")
                    continue
                del landmarkers
                keypoint = keypoint.reshape(-1)
                keypoint = apply_weighting_to_flattened(keypoint)
                keypoints.append(keypoint)
                break

        for i in range(len(keypoints)):
            keypoints[i] = keypoints[i].reshape(-1)

        X = np.array(keypoints)
        Y = np.array([self.label_list.index(label) for label in labels])
        return X, Y

    def __getitem__(self, index):
        indices = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X, Y = self.__data_generation(indices)
        return X, Y
    
class ValidationGenerator(keras.utils.Sequence):
    # X: keypoint files named 0.npy to 29.npy in each directory
    # Y: name of each directory
    def __init__(self, keypoint_path, label_list, batch_size, randomize=False):
        self.keypoint_path = keypoint_path
        self.label_list = label_list
        self.batch_size = batch_size
        self.randomize = randomize
        
        self.file_list = []
        self.labels = []

        for label in self.label_list:
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
        Y = np.array([self.label_list.index(label) for label in labels])
        # if self.randomize: print("---> Batch generated")
        return X, Y
    
    def __getitem__(self, index):
        inds = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X, Y = self.__data_generation(inds)
        return X, Y

if __name__ == "__main__":
    config = config_parser("config/config_image.yaml")

    
    # training_generator = DataGenerator(config.paths.dataset, 32, True)
    training_generator = ValidationGenerator(config.paths.keypoints, config.dictionary, 32, True)
    validation_generator = ValidationGenerator(config.paths.keypoints, config.dictionary, 32, False)

    # model type: multi-layer perceptron
    model = keras.models.Sequential([
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(len(config.dictionary), activation="softmax")
    ])

    # checkpoints
    checkpoint = keras.callbacks.ModelCheckpoint(config.paths.model_checkpoint, monitor="val_loss", save_best_only=True)

    # callbacks
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    # epoch_end = keras.callbacks.LambdaCallback(on_epoch_end=training_generator.on_epoch_end)


    model.compile(
        optimizer="adam", 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )

    model.fit(
        training_generator,
        epochs=128,
        shuffle=True,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stopping]
    )

    # model.fit(
    #     validation_generator,
    #     epochs=512
    # )
    model.save(config.paths.model)

    # print model train history
    print(model.history.history)

    # for each keypoint file, predict the label one by one and print the result
    # pred_dict = {}
    # print(config.dictionary)
    # for label in config.dictionary:
    #     preds = []
    #     for i in range(30):
    #         keypoint = np.load(os.path.join(config.paths.keypoints, label, f"{i}.npy"), mmap_mode="r")
    #         keypoint = keypoint.reshape(-1)
    #         keypoint = apply_weighting_to_flattened(keypoint)
    #         res = model.predict([np.array([keypoint])])[0]
    #         pred_class = np.argmax(res)
    #         pred_label = config.dictionary[pred_class]
    #         preds.append(pred_label)
    #     pred_dict[label] = preds
    # for label, preds in pred_dict.items():
    #     print(f"{label}: {' '.join(preds)}, category accuracy: {preds.count(label) / len(preds)}")
    # print(f"Overall accuracy: {sum([preds.count(label) for label, preds in pred_dict.items()]) / 30 / len(config.dictionary)}")

    # use the validation generator to predict the labels
    pred_dict = {}
    for i in range(len(validation_generator)):
        X, Y = validation_generator[i]
        res = model.predict(X)
        for j in range(len(Y)):
            pred_class = np.argmax(res[j])
            pred_label = config.dictionary[pred_class]
            if pred_label not in pred_dict:
                pred_dict[pred_label] = []
            pred_dict[pred_label].append(config.dictionary[Y[j]])
    for label, preds in pred_dict.items():
        print(f"{label}: {' '.join(preds)}, category accuracy: {preds.count(label) / len(preds)}")
    print(f"Overall accuracy: {sum([preds.count(label) for label, preds in pred_dict.items()]) / len(validation_generator) / 32}")

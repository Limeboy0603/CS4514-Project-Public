import os
os.environ['KERAS_BACKEND'] = "torch"
os.environ['KMP_DUPLICATE_LIB_OK'] = "True"
# os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
import keras
import numpy as np

from config import config_parser
from mp_util_legacy import preprocess_keypoints_multiple

np.random.seed(42)

class KeypointDataGenerator(keras.utils.Sequence):
    def __init__(self, keypoint_path, seq_max_len, batch_size=32, shuffle=False, transform=False):
        self.keypoint_path = keypoint_path
        self.seq_max_len = seq_max_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        
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
    
    def preprocess_keypoints(self, keypoints):
        angle = np.random.randint(-20, 20)
        tx = np.random.uniform(-0.4, 0.4)
        ty = np.random.uniform(-0.4, 0.4)
        scale = np.random.uniform(0.6, 1.2)
        return preprocess_keypoints_multiple(keypoints, angle=angle, tx=tx, ty=ty, scale=scale)

    def __data_generation(self, indexes):
        sequences = []
        labels = []

        for index in indexes:
            kp_file_name = self.file_list[index]
            label = self.labels[index]
            sequences.append(np.load(os.path.join(self.keypoint_path, label, kp_file_name), mmap_mode="r"))
            labels.append(label)

        for i in range(len(sequences)):
            if self.transform:
                sequences[i] = self.preprocess_keypoints(sequences[i])
            sequences[i] = sequences[i].reshape(self.seq_max_len, -1)

        X = np.array(sequences)
        Y = np.array([self.all_labels.index(label) for label in labels])
        return X, Y
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X, Y = self.__data_generation(indexes)
        return X, Y
    
def main(config_path: str):
    config = config_parser(config_path)

    training_generator = KeypointDataGenerator(
        config.paths.split_train, 
        seq_max_len=config.sequence.frame, 
        batch_size=32, 
        shuffle=True, 
        transform=True
    )
    validation_generator = KeypointDataGenerator(
        config.paths.split_val, 
        seq_max_len=config.sequence.frame, 
        batch_size=32, 
        shuffle=False, 
        transform=False
    )

    first_batch = training_generator.__getitem__(0)
    X_shape = first_batch[0].shape[1:]
    Y_shape = len(training_generator.all_labels)
    del first_batch

    model = Sequential()

    # 2 layers of LSTM with 256 units
    model.add(LSTM(1024, input_shape=X_shape, return_sequences=True, use_cudnn=False))
    model.add(LSTM(1024, return_sequences=False, use_cudnn=False))
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(512, activation="relu"))
    # model.add(Dropout(0.25))

    # model.add(LSTM(256, input_shape=X_shape, return_sequences=True))
    # model.add(LSTM(256, return_sequences=False))
    # model.add(Dense(256, activation="relu"))
    # model.add(Dense(128, activation="relu"))

    model.add(Dropout(0.25))
    model.add(Dense(Y_shape, activation="softmax"))

    model.summary()
    # model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate=0.0001), metrics=["accuracy"])

    model.fit(
        training_generator,
        # epochs=128,
        epochs=64,
        shuffle=True,
        validation_data=validation_generator,
        callbacks=[
            keras.callbacks.ModelCheckpoint(config.paths.model_checkpoint, monitor="val_loss", save_best_only=True),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
        ]
    )

    model.save(config.paths.model)
    # print(model.history.history)

    # multilayer: 0.9844
    

if __name__ == "__main__":
    main("config/config_clip.yaml" if os.name == "nt" else "config/config_clip_linux.yaml")
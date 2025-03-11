import os
os.environ['KERAS_BACKEND'] = 'torch'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import keras
from config import config_parser

def main(config_path: str):
    # config = config_parser("config/config_clip.yaml")
    config = config_parser(config_path)
    model = keras.models.load_model(config.paths.model)

    # validation set
    pred_dict = {}
    dictionary = sorted(os.listdir(config.paths.split_val))
    for label in dictionary:
        preds = []
        label_path = os.path.join(config.paths.split_val, label)
        for file in os.listdir(label_path):
            kp = np.load(os.path.join(label_path, file))
            pred_kp = kp.reshape(1, kp.shape[0], -1)
            pred = model.predict(pred_kp)
            pred = np.argmax(pred)
            pred = dictionary[pred]
            preds.append(pred)
        pred_dict[label] = preds
    print("Using path:", config.paths.split_val)
    print(pred_dict)

    correct = 0
    total = 0
    for key in pred_dict:
        for pred in pred_dict[key]:
            if pred == key:
                correct += 1
            total += 1
    print("Accuracy: ", correct / total)

    # testing set
    pred_dict = {}
    dictionary = sorted(os.listdir(config.paths.split_test))
    for label in dictionary:
        preds = []
        label_path = os.path.join(config.paths.split_test, label)
        for file in os.listdir(label_path):
            kp = np.load(os.path.join(label_path, file))
            pred_kp = kp.reshape(1, kp.shape[0], -1)
            pred = model.predict(pred_kp)
            pred = np.argmax(pred)
            pred = dictionary[pred]
            preds.append(pred)
        pred_dict[label] = preds
    print("Using path:", config.paths.split_test)
    print(pred_dict)

    # compute accuracy
    correct = 0
    total = 0
    for key in pred_dict:
        for pred in pred_dict[key]:
            if pred == key:
                correct += 1
            total += 1
    print("Accuracy: ", correct / total)

    # single-layer: 0.9666666666666667
    # multi-layer: 0.9888888888888889

if __name__ == "__main__":
    main("config/config_clip.yaml" if os.name == "nt" else "config/config_clip_linux.yaml")
from tvb_hksl_split_parser import tvb_hksl_split_parser
from src_old.seq2seq.keypoint_preprocessing import get_provided_keypoints
import numpy as np
import json
import os
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pandas as pd

DEV_MODE = False

if __name__ == "__main__":
    keypoint_dir = r"E:\dataset\tvb-hksl-news\keypoints"

    train_file = ""
    word_dict_path = ""
    model_path = ""
    result_save_path = ""

    if DEV_MODE:
        train_file = r"E:\dataset\tvb-hksl-news\split\dev.csv"
        word_dict_path = "data/word_dict_dev.json"
        model_path = "models/tvb_hksl_dev_2.keras"
        result_save_path = "data/predicted_dev_2.csv"
    else:
        train_file = r"E:\dataset\tvb-hksl-news\split\train.csv"
        word_dict_path = "data/word_dict_train.json"
        model_path = "models/tvb_hksl_train_2.keras"
        result_save_path = "data/predicted_train_2.csv"
    
    train_parser = tvb_hksl_split_parser(train_file)

    test_file = r"E:\dataset\tvb-hksl-news\split\test.csv"
    test_parser = tvb_hksl_split_parser(test_file)

    X_train, Y_train, X_test, Y_test, word_dict = get_provided_keypoints(train_parser, test_parser, keypoint_dir) 
    # this time, read the word_dict from the file instead of generating it
    with open(word_dict_path, "r") as f:
        word_dict = json.load(f)
        
    print(word_dict)
    # load the model
    model = load_model(model_path)

    
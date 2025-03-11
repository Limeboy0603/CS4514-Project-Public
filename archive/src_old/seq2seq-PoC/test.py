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

    # get accuracy
    accuracy = model.evaluate([X_test, Y_test], Y_test)

    # predict the test set
    Y_pred = model.predict(X_test)
    
    # convert the predicted output to actual words
    Y_pred_words = []
    for i in range(len(Y_pred)):
        Y_pred_words.append([list(word_dict.keys())[list(word_dict.values()).index(np.argmax(word))] for word in Y_pred[i]])
    Y_pred_words = np.array(Y_pred_words)

    # align Y_pred_words with Y_test
    Y_test_words = []
    for i in range(len(Y_test)):
        Y_test_words.append([list(word_dict.keys())[list(word_dict.values()).index(word)] for word in Y_test[i]])
    Y_test_words = np.array(Y_test_words)

    # for each item in Y_pred_words and Y_test_words, change list of strings to a single string
    # remove all the <PAD> values
    Y_pred_words = [" ".join([word for word in sequence if word != "<PAD>"]) for sequence in Y_pred_words]
    Y_test_words = [" ".join([word for word in sequence if word != "<PAD>"]) for sequence in Y_test_words]

    Y_accuracy = []
    for pred, test in zip(Y_pred_words, Y_test_words):
        pred_words = pred.split()
        test_words = test.split()
        # Calculate the number of matching words and divide by the length of the test sentence
        accuracy = sum(1 for pred_word, test_word in zip(pred_words, test_words) if pred_word == test_word) / len(test_words) if test_words else 0
        Y_accuracy.append(accuracy)

    # export the predicted words to a csv file
    df = pd.DataFrame()
    df["Y_test"] = Y_test_words
    df["Y_pred"] = Y_pred_words
    df["accuracy"] = Y_accuracy

    df.to_csv(result_save_path, index=False)
import pandas as pd
import numpy as np
import json
from keras.models import load_model # type: ignore
from keras.backend import clear_session # type: ignore
import os
from config_object import ConfigObject
from tvb_hksl_split_parser import tvb_hksl_split_parser
from keypoint_generator import KeypointGenerator
import argparse

if __name__ == "__main__":
    clear_session()
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/wsl/provided_train.yaml")
    args = parser.parse_args()
    CONFIG_PATH = args.config

    config_module = ConfigObject(CONFIG_PATH)
    word_dict_path = config_module.get_word_dict_path()
    train_file = config_module.get_train_file()
    test_file = config_module.get_test_file()
    keypoint_dir = config_module.get_keypoint_dir()
    cache_dir = config_module.get_cache_dir()

    train_parser = tvb_hksl_split_parser(train_file)
    test_parser = tvb_hksl_split_parser(test_file)

    keypoint_generator = KeypointGenerator(train_parser, test_parser, keypoint_dir, cache_dir, True)

    keypoint_obj = keypoint_generator.get_keypoints(config_module.get_type())

    model = load_model(config_module.get_model_path())
    encoder_model = load_model(config_module.get_encoder_path())
    decoder_model = load_model(config_module.get_decoder_path())

    X_test, Y_test = keypoint_obj.get_test_data()
    word_dict = keypoint_obj.get_word_dict()

    reverse_word_dict = {v: k for k, v in word_dict.items()}

    # test the model with ground truth
    decoder_input_data_test, decoder_target_data_test = keypoint_obj.get_decoder_test_data()
    accuracy = model.evaluate([X_test, decoder_input_data_test], decoder_target_data_test)
    print(accuracy)

    # export the prediction results for all samples in the test set to a csv file
    Y_pred = model.predict([X_test, decoder_input_data_test])
    # print the first prediction
    print(Y_pred[0])
    # print the decoded first prediction
    print([reverse_word_dict[np.argmax(i)] for i in Y_pred[0]])

    Y_pred_words = []
    # convert from one-hot encoding to actual words
    for i in range(len(Y_pred)):
        Y_pred_words.append([reverse_word_dict[np.argmax(word)] for word in Y_pred[i]])
    Y_pred_words = np.array(Y_pred_words)

    Y_test_words = []
    for i in range(len(Y_test)):
        Y_test_words.append([reverse_word_dict[np.argmax(word)] for word in Y_test[i]])
    Y_test_words = np.array(Y_test_words)

    Y_pred_words = [" ".join(i) for i in Y_pred_words]
    Y_test_words = [" ".join(i) for i in Y_test_words]

    df = pd.DataFrame(data={"predicted": Y_pred_words, "actual": Y_test_words})
    df.to_csv("test.csv", index=False)

    exit()

    # inference
    
    def decode_sequence(input_seq):
        states_value = encoder_model.predict(input_seq) 

        target_seq = np.zeros((1, 1, len(word_dict)))
        target_seq[0, 0, word_dict["<START>"]] = 1

        stop_condition = False
        decoded_sentence = ""

        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_word_dict[sampled_token_index]
            decoded_sentence += " " + sampled_char

            if sampled_char == "<END>" or len(decoded_sentence) > 100:
                stop_condition = True

            target_seq = np.zeros((1, 1, len(word_dict)))
            target_seq[0, 0, sampled_token_index] = 1

            states_value = [h, c]

        return decoded_sentence
    
    response = decode_sequence(X_test[0:1])
    print(response)
    actual = [reverse_word_dict[np.argmax(i)] for i in Y_test[0]]
    print(actual)

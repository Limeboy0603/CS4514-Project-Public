import tensorflow as tf
from keras.layers import Dense, LSTM, Input, Masking, Dropout # type: ignore
from keras.models import Model # type: ignore
from keras.callbacks import ModelCheckpoint # type: ignore
from keras.backend import clear_session # type: ignore
from keras.optimizers import RMSprop # type: ignore
from tvb_hksl_split_parser import tvb_hksl_split_parser
from keypoint_generator import KeypointGenerator
from config_object import ConfigObject
import numpy as np
import argparse
import os

EPOCH = 1000
BATCH_SIZE = 64
LATENT_DIM = 256

if __name__ == "__main__":
    clear_session()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/wsl/provided_train.yaml")
    parser.add_argument("--memmap", type=bool, default=True)
    parser.add_argument("--gpu", type=bool, default=False)
    args = parser.parse_args()
    CONFIG_PATH = args.config
    MEMMAP = args.memmap
    GPU = args.gpu

    if not GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    else:
        # reset the environment variable
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        physical_devices = tf.config.list_physical_devices('GPU')
        print(physical_devices)

    # exit()

    config_module = ConfigObject(CONFIG_PATH)
    kp_type = config_module.get_type()
    keypoint_dir = config_module.get_keypoint_dir()
    train_file = config_module.get_train_file()
    model_path = config_module.get_model_path()
    checkpoint_path = config_module.get_checkpoint_path()
    test_file = config_module.get_test_file()
    cache_dir = config_module.get_cache_dir()

    train_parser = tvb_hksl_split_parser(train_file)
    test_parser = tvb_hksl_split_parser(test_file)

    keypoint_generator = KeypointGenerator(train_parser, test_parser, keypoint_dir, cache_dir, MEMMAP)

    # should automatically load cache if it exists
    keypoints_obj = keypoint_generator.get_keypoints(kp_type)

    word_dict = keypoints_obj.get_word_dict()
    X_train, Y_train = keypoints_obj.get_train_data()
    X_test, Y_test = keypoints_obj.get_test_data()
    decoder_input_data_train, decoder_target_data_train = keypoints_obj.get_decoder_train_data()
    decoder_input_data_test, decoder_target_data_test = keypoints_obj.get_decoder_test_data()

    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)
    print("decoder_input_data_train shape:", decoder_input_data_train.shape)
    print("decoder_target_data_train shape:", decoder_target_data_train.shape)
    print("decoder_input_data_test shape:", decoder_input_data_test.shape)
    print("decoder_target_data_test shape:", decoder_target_data_test.shape)

    print(decoder_input_data_test)
    print(decoder_target_data_test)

    # print the decoded input and target data for the first sample, they are one-hot encoded
    # turn them back to the original word
    decoder_input_sample = decoder_input_data_train[0]
    decoder_target_sample = decoder_target_data_train[0]
    # print(decoder_input_sample)
    # print(decoder_target_sample)
    for i in range(decoder_input_sample.shape[0]):
        if np.sum(decoder_input_sample[i]) > 0:
            print(list(word_dict.keys())[list(word_dict.values()).index(np.argmax(decoder_input_sample[i]))], end=" ")
    print("\n")
    for i in range(decoder_target_sample.shape[0]):
        if np.sum(decoder_target_sample[i]) > 0:
            print(list(word_dict.keys())[list(word_dict.values()).index(np.argmax(decoder_target_sample[i]))], end=" ")
    print("\n")
    # exit()

    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)

    # DEBUG: export the first sample of X_train to a csv file
    # np.savetxt("X_train.csv", X_train[0], delimiter=",")

    # throws error if NaN is found
    assert not np.isnan(X_train).any()
    assert not np.isnan(decoder_input_data_train).any()
    assert not np.isnan(decoder_target_data_train).any()

    print("Data check passed - no NaN found in the input data")
     
    # exit()

    print("Starting model training...")

    """
    Acceptable input shape for the model:
    X_train: (batch_size, timesteps, features)
    - batch_size: number of samples in the batch
    - timesteps: number of frames in the video
    - features: number of keypoints in each frame, same length for all frames, flattened from 2 dimensions to 1

    Y_train: (batch_size, sequence_len, class_size)
    - batch_size: number of samples in the batch
    - sequence_len: number of words in the sentence
    - class_size: number of classes in the word_dict, each word in the sentence is one-hot encoded
    """

    # Encoder
    encoder_input = Input(shape=(None, X_train.shape[2]))
    encoder_mask = Masking()(encoder_input)
    encoder = LSTM(LATENT_DIM, return_state=True)
    # encoder_output, state_h, state_c = encoder(encoder_mask)
    encoder_output, state_h, state_c = encoder(encoder_input)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_input = Input(shape=(None, len(word_dict)))
    decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)

    # Dense    
    decoder_dense = Dense(len(word_dict), activation="softmax")
    decoder_output = decoder_dense(decoder_outputs)

    model = Model([encoder_input, decoder_input], decoder_output)
    model.compile(optimizer=RMSprop(), loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    model.summary()
    # exit()

    # model_checkpoint = ModelCheckpoint(
    #     checkpoint_path, 
    #     save_weights_only=False,
    #     # monitor="val_accuracy",
    #     monitor="val_categorical_accuracy",
    #     mode="max",
    #     save_best_only=True
    # )

    model.fit(
        [X_train, decoder_input_data_train],
        decoder_target_data_train,
        epochs=EPOCH, 
        batch_size=BATCH_SIZE, 
        # validation_data=validation,
        # callbacks=[model_checkpoint]
    )
    model.save(model_path)

    # reset the environment variable regardless
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    # immediately evaluate the model after training with ground truth
    accuracy = model.evaluate([X_test, decoder_input_data_test], decoder_target_data_test)
    print(accuracy)

    # export the prediction results for all samples in the test set to a csv file
    Y_pred = model.predict([X_test, decoder_input_data_test])
    # print the first prediction
    print(Y_pred[0])

    # convert from one-hot encoding to actual words
    Y_pred_words = []
    for i in range(len(Y_pred)):
        Y_pred_words.append([list(word_dict.keys())[list(word_dict.values()).index(np.argmax(word))] for word in Y_pred[i]])
    Y_pred_words = np.array(Y_pred_words)

    Y_test_words = []
    for i in range(len(decoder_target_data_test)):
        Y_test_words.append([list(word_dict.keys())[list(word_dict.values()).index(np.argmax(word))] for word in decoder_target_data_test[i]])
    Y_test_words = np.array(Y_test_words)

    Y_pred_words = [" ".join(i) for i in Y_pred_words]
    Y_test_words = [" ".join(i) for i in Y_test_words]

    import pandas as pd
    df = pd.DataFrame(data={"predicted": Y_pred_words, "actual": Y_test_words})
    df.to_csv("test.csv", index=False)

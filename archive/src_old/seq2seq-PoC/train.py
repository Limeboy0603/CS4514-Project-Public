from keras.layers import Dense, LSTM, Embedding, Input, Masking, TimeDistributed, Reshape # type: ignore
from keras.models import Model # type: ignore
from keras.callbacks import ModelCheckpoint # type: ignore
import os
import json
from tvb_hksl_split_parser import tvb_hksl_split_parser
from src_old.seq2seq.keypoint_preprocessing import get_provided_keypoints, get_mediapipe_keypoints

"""
This script attempts to create a Seq2Seq model for the tvb-hksl dataset.

The model is based on the following paper:
Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. 
Advances in neural information processing systems, 27.
"""
DEV_MODE = True
KEYPOINT_MODE = "provided" # "provided" or "mediapipe"

if __name__ == "__main__":
    train_file = ""
    word_dict_path = ""
    model_path = ""
    checkpoint_path = ""
    set_epoch = 0
    set_batch_size = 0

    if KEYPOINT_MODE == "provided":
        keypoint_dir = r"E:\dataset\tvb-hksl-news\keypoints"
        if DEV_MODE:
            word_dict_path = "data/word_dict_dev.json"
            model_path = "models/tvb_hksl_dev_2.keras"
            checkpoint_path = "models/checkpoint/tvb_hksl_dev_2.keras"
        else:
            word_dict_path = "data/word_dict_train.json"
            model_path = "models/tvb_hksl_train_2.keras"
            checkpoint_path = "models/checkpoint/tvb_hksl_train_2.keras"
    elif KEYPOINT_MODE == "mediapipe":
        keypoint_dir = r"E:\dataset\tvb-hksl-news\keypoints_mediapipe"
        if DEV_MODE:
            word_dict_path = "data/word_dict_dev_mediapipe.json"
            model_path = "models/tvb_hksl_dev_mediapipe_xyz_2.keras"
            checkpoint_path = "models/checkpoint/tvb_hksl_dev_mediapipe_xyz_2.keras"
        else:
            word_dict_path = "data/word_dict_train_mediapipe.json"
            model_path = "models/tvb_hksl_train_mediapipe_xyz_2.keras"
            checkpoint_path = "models/checkpoint/tvb_hksl_train_mediapipe_xyz_2.keras"
            # TODO: later test without z-axis
            # checkpoint_path = "models/checkpoint/tvb_hksl_train_mediapipe_xy.keras"
    if DEV_MODE:
        train_file = r"E:\dataset\tvb-hksl-news\split\dev.csv"
        set_epoch = 100
        set_batch_size = 64
    else:
        train_file = r"E:\dataset\tvb-hksl-news\split\train.csv"
        set_epoch = 50
        set_batch_size = 64
    test_file = r"E:\dataset\tvb-hksl-news\split\test.csv"

    train_parser = tvb_hksl_split_parser(train_file)
    test_parser = tvb_hksl_split_parser(test_file)

    if KEYPOINT_MODE == "provided":
        X_train, Y_train, X_test, Y_test, word_dict = get_provided_keypoints(train_parser, test_parser, keypoint_dir)
    else: X_train, Y_train, X_test, Y_test, word_dict = get_mediapipe_keypoints(train_parser, test_parser, keypoint_dir)
    # exit()
    with open(word_dict_path, "w") as f:
        json.dump(word_dict, f, indent=4)

    """
    Acceptable input shape for the model:
    X_train: (batch_size, timesteps, features)
    - batch_size: number of samples in the batch
    - timesteps: number of frames in the video
    - features: number of keypoints in each frame, must be 1D and same length for all frames

    Y_train: (batch_size, sequence_len, class_size)
    - batch_size: number of samples in the batch
    - sequence_len: number of words in the sentence
    - class_size: number of classes in the word_dict
    """

    # Encoder
    encoder_input = Input(shape=(X_train.shape[1], X_train.shape[2]))
    encoder = LSTM(1024, return_state=True)
    encoder_output, state_h, state_c = encoder(encoder_input)
    encoder_states = [state_h, state_c]

    # Decoder
    # attempt: skip the masking layer for TFLite conversion
    # decoder_input = Input(shape=(Y_train.shape[1],)) 
    decoder_input = Masking(mask_value=word_dict["<PAD>"])(Input(shape=(Y_train.shape[1],)))
    decoder_embedding = Embedding(len(word_dict), 1024)
    decoder_input_embedding = decoder_embedding(decoder_input)
    decoder_lstm = LSTM(1024, return_sequences=True, return_state=True)

    # attempt: skip the embedding layer for TFLite conversion
    # decoder_output, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states) 
    decoder_output, _, _ = decoder_lstm(decoder_input_embedding, initial_state=encoder_states)
    decoder_dense = Dense(len(word_dict), activation="softmax")
    decoder_output = decoder_dense(decoder_output)
    # actual_output = Lambda(lambda x: tf.argmax(x, axis=-1))(decoder_output)

    model = Model([encoder_input, decoder_input], decoder_output)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # model.compile(optimizer="adam", loss="squared_hinge", metrics=["accuracy"])
    model.summary()
    # exit()

    model_checkpoint = ModelCheckpoint(
        checkpoint_path, 
        save_weights_only=False,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True
    )

    model.fit([X_train, Y_train], Y_train, epochs=set_epoch, batch_size=set_batch_size, validation_data=([X_test, Y_test], Y_test), callbacks=[model_checkpoint])
    model.save(model_path)
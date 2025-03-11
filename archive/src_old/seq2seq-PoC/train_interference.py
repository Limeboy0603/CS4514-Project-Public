from keras.layers import Dense, LSTM, Embedding, Input, Masking # type: ignore
from keras.models import Model, load_model # type: ignore
from keras.callbacks import ModelCheckpoint # type: ignore
import os
import json
from tvb_hksl_split_parser import tvb_hksl_split_parser
from src_old.seq2seq.keypoint_preprocessing import get_provided_keypoints, get_mediapipe_keypoints
import numpy as np

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
            model_path = "models/tvb_hksl_dev_interference.keras"
            checkpoint_path = "models/checkpoint/tvb_hksl_dev_interference.keras"
        else:
            word_dict_path = "data/word_dict_train.json"
            model_path = "models/tvb_hksl_train_interference.keras"
            checkpoint_path = "models/checkpoint/tvb_hksl_train_interference.keras"
    elif KEYPOINT_MODE == "mediapipe":
        keypoint_dir = r"E:\dataset\tvb-hksl-news\keypoints_mediapipe"
        if DEV_MODE:
            word_dict_path = "data/word_dict_dev_mediapipe.json"
            model_path = "models/tvb_hksl_dev_mediapipe_xyz_interference.keras"
            checkpoint_path = "models/checkpoint/tvb_hksl_dev_mediapipe_xyz_interference.keras"
        else:
            word_dict_path = "data/word_dict_train_mediapipe.json"
            model_path = "models/tvb_hksl_train_mediapipe_xyz_interference.keras"
            checkpoint_path = "models/checkpoint/tvb_hksl_train_mediapipe_xyz_interference.keras"
            # TODO: later test without z-axis
            # checkpoint_path = "models/checkpoint/tvb_hksl_train_mediapipe_xy.keras"
    if DEV_MODE:
        train_file = r"E:\dataset\tvb-hksl-news\split\dev.csv"
        set_epoch = 50
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
    
    word_dict["<START>"] = len(word_dict)
    with open(word_dict_path, "w") as f:
        json.dump(word_dict, f, indent=4)
    
    # append the <START> token to each sample in Y_train and Y_test
    Y_train = np.insert(Y_train, 0, word_dict["<START>"], axis=1)
    Y_test = np.insert(Y_test, 0, word_dict["<START>"], axis=1)

    """
    Acceptable input shape for the model:
    X_train: (batch_size, timesteps, features)
    - batch_size: number of samples in the batch
    - timesteps: number of frames in the video
    - features: number of keypoints in each frame, must be 1D and same length for all frames

    Y_train: (batch_size, timesteps)
    - batch_size: number of samples in the batch
    - timesteps: number of words in the glosses
    """

    # Encoder
    encoder_input = Input(shape=(X_train.shape[1], X_train.shape[2]))
    encoder = LSTM(256, return_state=True)
    encoder_output, state_h, state_c = encoder(encoder_input)
    encoder_states = [state_h, state_c]

    # Decoder
    # attempt: skip the masking layer for TFLite conversion
    # decoder_input = Input(shape=(Y_train.shape[1],)) 
    decoder_input = Masking(mask_value=word_dict["<PAD>"])(Input(shape=(Y_train.shape[1],)))
    decoder_embedding = Embedding(len(word_dict), 256)
    decoder_input_embedding = decoder_embedding(decoder_input)
    decoder_lstm = LSTM(256, return_sequences=True, return_state=True)

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

    model = load_model(model_path)

    # Define the encoder model for inference
    encoder_model = Model(encoder_input, encoder_states)

    # Define the decoder model for inference
    decoder_state_input_h = Input(shape=(256,), name='input_3')
    decoder_state_input_c = Input(shape=(256,), name='input_4')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_input_single = Input(shape=(1,))
    decoder_input_single_embedding = decoder_embedding(decoder_input_single)
    decoder_output, state_h, state_c = decoder_lstm(decoder_input_single_embedding, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_output = decoder_dense(decoder_output)

    decoder_model = Model([decoder_input_single] + decoder_states_inputs, [decoder_output] + decoder_states)

    # inference function
    # Inference function
    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))
        # Populate the first token of target sequence with the start token.
        target_seq[0, 0] = word_dict["<START>"]

        # Sampling loop for a batch of sequences
        stop_condition = False
        decode_sequence = []
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            # sampled_char = reverse_word_dict[sampled_token_index]
            sampled_char = list(word_dict.keys())[list(word_dict.values()).index(sampled_token_index)]
            decoded_sentence += ' ' + sampled_char
            decode_sequence.append(sampled_char)

            # Exit condition: either hit max length or find stop token.
            if (len(decode_sequence) > Y_train.shape[1]):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Update states
            states_value = [h, c]

        return decoded_sentence

    # Example usage
    input_seq = X_test[0:1]
    decoded_sentence = decode_sequence(input_seq)
    print("Decoded sentence:", decoded_sentence)

    # Save the inference models
    encoder_model.save('encoder_model.h5')
    decoder_model.save('decoder_model.h5')
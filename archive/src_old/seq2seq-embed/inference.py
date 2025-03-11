from keras.models import Model # type: ignore
from keras.layers import Input, LSTM, Dense, Masking # type: ignore
from keras.models import load_model # type: ignore
from keras.backend import clear_session # type: ignore
from config_object import ConfigObject
import json
import argparse
import os

LATENT_DIM = 1024

if __name__ == "__main__":
    clear_session()
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/wsl/provided_train.yaml")
    args = parser.parse_args()
    CONFIG_PATH = args.config

    config_module = ConfigObject(CONFIG_PATH)
    model_path = config_module.get_model_path()
    encoder_path = config_module.get_encoder_path()
    decoder_path = config_module.get_decoder_path()
    word_dict_path = config_module.get_word_dict_path()

    current_model = load_model(model_path)
    print(current_model.summary())
    print(current_model.input_shape)
    # exit()
    with open(word_dict_path, "r") as f:
        word_dict = json.load(f)

    # copy the entire model architecture from train.py, and manually load the weights

    # TODO: rewrite the entire inference code. base on https://keras.io/examples/nlp/lstm_seq2seq/

    # Encoder
    encoder_input = Input(shape=(None, current_model.input_shape[0][2]))
    encoder_mask = Masking(mask_value=0.0)(encoder_input)
    encoder = LSTM(LATENT_DIM, return_state=True, use_cudnn=False)
    encoder_output, state_h, state_c = encoder(encoder_mask)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_input = Input(shape=(None, len(word_dict)))
    decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True, use_cudnn=False)
    decoder_outputs, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)

    # Dense
    decoder_dense = Dense(len(word_dict), activation="softmax")
    decoder_output = decoder_dense(decoder_outputs)

    model = Model([encoder_input, decoder_input], decoder_output)
    model.set_weights(current_model.get_weights())

    # Inference Encoder
    encoder_model = Model(encoder_input, encoder_states)
    encoder_model.summary()
    encoder_model.save(encoder_path)

    # Inference Decoder
    decoder_state_input_h = Input(shape=(LATENT_DIM,))
    decoder_state_input_c = Input(shape=(LATENT_DIM,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_output, state_h, state_c = decoder_lstm(decoder_input, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    decoder_dense = Dense(len(word_dict), activation="softmax")
    decoder_output = decoder_dense(decoder_output)

    decoder_model = Model(
        [decoder_input] + decoder_states_inputs, 
        [decoder_output] + decoder_states
    )
    decoder_model.summary()
    decoder_model.save(decoder_path)
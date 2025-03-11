# This script is designed to train the model on server without Jupyter Notebook
# Assumes that you have already cached all training data
# Run this script in ./src directory

# Copy all of your config variables from the Jupyter Notebook here

MODE = "train" # train |dev

# set to True if you want to cache Y values extracted from the split file
CACHE_Y = True

# set to True if you want to cache the batched data. 
CACHE_BATCH = False

# set to True if you want to use cached batch data to train. this will cause every epoch to train from the same data
# if you wish to train from transformed data every epoch, set this to False. this will replace the data input of model fitting process with a generator
USE_CACHE_BATCH = True

# set to True if you want to apply weighting while generating
GENERATE_WEIGHT = False

# set to True if you want to apply weighting to cache data
USE_WEIGHT = True

# model parameters config
BATCH_SIZE = 32
EPOCH = 256
LATENT_DIM = 512

weighted_suffix = "weighted" if GENERATE_WEIGHT or USE_WEIGHT else ""
MODEL_PATH = f"../model/{MODE}_model_{weighted_suffix}_{BATCH_SIZE}_{EPOCH}_{LATENT_DIM}.keras"
ENCODER_PATH = f"../model/{MODE}_encoder_{weighted_suffix}_{BATCH_SIZE}_{EPOCH}_{LATENT_DIM}.keras"
DECODER_PATH = f"../model/{MODE}_decoder_{weighted_suffix}_{BATCH_SIZE}_{EPOCH}_{LATENT_DIM}.keras"

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] ="3"

CACHE_DIR = f"../cache/{MODE}"
os.makedirs(CACHE_DIR, exist_ok=True)
RESULT_FILE_NAME = f"../results/{MODE}_results.csv"

import pandas as pd
import numpy as np
import json
import cv2
from mediapipe.python.solutions.holistic import Holistic
import time
import keras
from concurrent.futures import ThreadPoolExecutor
from keras.optimizers import RMSprop
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Masking
import csv
import tensorflow as tf
from functools import partial

# check for cuda availability
assert len(tf.config.list_physical_devices('GPU')) > 0, "No GPU available"

# check whether the script is running on tmux
assert os.environ.get("TMUX") is not None, "This script should be run on tmux"

with open("../data/weighting_list.json", "r") as f:
    weighting_list = json.load(f)
with open("../data/word_dict.json", "r") as f:
    word_dict = json.load(f)

x_dir = os.path.join(CACHE_DIR, "x")
decoder_input_dir = os.path.join(CACHE_DIR, "decoder_input")
decoder_target_dir = os.path.join(CACHE_DIR, "decoder_target")

class CachedKeypointGenerator(keras.utils.Sequence):
    def __init__(self, x_dir, decoder_input_dir, decoder_target_dir, batch_size=32):
        self.x_dir = x_dir
        self.decoder_input_dir = decoder_input_dir
        self.decoder_target_dir = decoder_target_dir
        self.batch_size = batch_size
        self.list_x_files = sorted(os.listdir(self.x_dir))
        self.list_decoder_input_files = sorted(os.listdir(self.decoder_input_dir))
        self.list_decoder_target_files = sorted(os.listdir(self.decoder_target_dir))
        self.total_files = len(self.list_x_files)
        self.batch_index = 0

    def __len__(self):
        return self.total_files

    def __getitem__(self, idx):
        batch_x = np.load(os.path.join(self.x_dir, self.list_x_files[idx]), mmap_mode="r")
        batch_decoder_input = np.load(os.path.join(self.decoder_input_dir, self.list_decoder_input_files[idx]), mmap_mode="r")
        batch_decoder_target = np.load(os.path.join(self.decoder_target_dir, self.list_decoder_target_files[idx]), mmap_mode="r")
        if USE_WEIGHT:
            # on the decoder input and target, apply the weighting list
            batch_decoder_input = batch_decoder_input * weighting_list
            batch_decoder_target = batch_decoder_target * weighting_list
        return (batch_x, batch_decoder_input), batch_decoder_target
keypoint_generator = CachedKeypointGenerator(x_dir, decoder_input_dir, decoder_target_dir, batch_size=BATCH_SIZE)


(temp_batch_x, temp_batch_decoder_input), temp_batch_decoder_target = keypoint_generator.__getitem__(0)
len_features = len(temp_batch_x[0][0])

# encoder
encoder_input = Input(shape=(None, len_features))
encoder_mask = Masking()(encoder_input)
encoder_lstm = LSTM(LATENT_DIM, return_state=True, return_sequences=True, use_cudnn=False)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_mask)
encoder_states = [state_h, state_c]

# decoder
decoder_input = Input(shape=(None, len(word_dict)))
decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True, use_cudnn=False)
decoder_outputs, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)
decoder_dense = Dense(len(word_dict), activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_input, decoder_input], decoder_outputs)
model.compile(optimizer=RMSprop(), loss="categorical_crossentropy", metrics=["categorical_accuracy"])
model.summary()

model.fit(keypoint_generator, epochs=EPOCH)
model.save(MODEL_PATH)

# Inference models
encoder_model = Model(encoder_input, encoder_states)

decoder_state_input_h = Input(shape=(LATENT_DIM,))
decoder_state_input_c = Input(shape=(LATENT_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_input, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_input] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# save the encoder and decoder models
encoder_model.save(ENCODER_PATH)
decoder_model.save(DECODER_PATH)
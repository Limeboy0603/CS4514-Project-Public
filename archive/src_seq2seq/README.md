# Sign Language Recognition and Translation using Motion Tracking and Deep Learning

This project aims to develop a system that translates continuous sign languages to natural sentences.

## Running Each Individual Part

### Part 1: Sign Language to Sequence of Glosses

- You should first create a virtual environment using `requirements.txt` in this directory
- Feature extraction takes a VERY long time to perform. If you have enough time to attempt to replicate this, you may either:
    - Set `CACHE_BATCH` and `USE_CACHE_BATCH` to `True` in the following cell, and this notebook will cache all the keypoints; or
    - Set `USE_CACHE_BATCH` to `False` in the following cell, and features will be extracted on the fly (takes very long!); or
    - (CPU) Run everything before `Feature Extraction` in this notebook, then change the setting at the top of `keypoint-gen-cpu.py` and run the script; or
    - (GPU) Run everything before `Feature Extraction` in this notebook, then change the setting at the top of `keypoint-gen-gpu.py` and run the script **on Ubuntu with GPU**. **Windows, MacOS, WSL Ubuntu, or any other Linux distributions are NOT supported** 
        - Since Holistic does not support GPU, this script uses the latest solution on face, pose and hand landmarker which includes 10 more face keypoints for the iris.
        - Since the face contours does not utilize these 10 keypoints, they will all be filtered during generation.
        - Thus, generated keypoints are also compatible with the intended shape of the model.

1. Create a virtual environment using `src_sign2gloss/requirements.txt`
```sh
mkdir .venv
python3 -m venv .venv/sign2gloss
source .venv/sign2gloss/bin/activate
pip install -r src_sign2gloss/requirements.txt
```
2. Change the settings in the second cell of the Jupyter Notebook `src_sign2gloss/play_seq2seq.ipynb`
    - I **highly** recommend that you read the first cell regarding data generation.
3. Run the entire Jupyter Notebook in the source directory
```sh
cd src_sign2gloss 
jupyter execute play_seq2seq.ipynb
```

This will create the main model, encoder model and decoder model in `./model` with a subdirectory named based on your settings.

This will also generate prediction results on the data in `./results` with a subdirectory named based on your settings.

### Part 2: Sequence of Glosses to Natural Sentences

**This part is only for testing the usage of LLMs.**

1. Create a virtual environment using `src_gloss2sentence/requirements.txt`
```sh
mkdir .venv
python3 -m venv .venv/gloss2sentence
source .venv/gloss2sentence/bin/activate
pip install -r gloss2sentence/requirements.txt
```
2. Play around with `gloss2sentence/nlp-test.py`. Really, there is nothing much to say.

## Deployment and Demonstration

1. Create a new `tmux` environment and start the flask app `src_gloss2sentence/deploy.py`
```sh
tmux

# in tmux
source .venv/gloss2sentence/bin/activate
flask --app src_gloss2sentence/deploy.py run -p 5001
# ^B-D to leave tmux
```

Alternatively, start it's associated Docker container
```sh
docker compose up -d gloss2sentence
```

2. Run `src_sign2gloss/demo.py`. See options below.
```
options:
usage: demo.py [-h] [--use_gui] [--encoder_path ENCODER_PATH] [--decoder_path DECODER_PATH] [--input_path INPUT_PATH] [--random_angle RANDOM_ANGLE] [--random_tx RANDOM_TX] [--random_ty RANDOM_TY] [--random_scale RANDOM_SCALE]

options:
  -h, --help            show this help message and exit
  --use_gui             Use GUI for file selection
  --encoder_path ENCODER_PATH
                        Path to the encoder model. Default: 'model/train_LSTM_weighted_32_512_1024_1/encoder.keras'
  --decoder_path DECODER_PATH
                        Path to the decoder model. Default: 'model/train_LSTM_weighted_32_512_1024_1/decoder.keras'
  --input_path INPUT_PATH
                        Path to the folder containing frames. Default: 'dataset/tvb-hksl-news/frames/2020-03-14/002832-003033'
  --random_angle RANDOM_ANGLE
                        Random angle for image transformation. Default: 0
  --random_tx RANDOM_TX
                        Random translation in x-axis for image transformation. Default: 0
  --random_ty RANDOM_TY
                        Random translation in y-axis for image transformation. Default: 0
  --random_scale RANDOM_SCALE
                        Random scaling for image transformation. Default: 1
```

How it works:
1. Generate keypoints using mediapipe.
2. Sends the keypoints to encoder.
3. Decoder receives state values from encoder and generate a sequence of glosses.
4. Sends the glosses to the running flask API to obtain the natural sentence.
5. Outputs the sequence of glosses and the natural sentence.

## Roadmap

- Sign to Gloss (Done?)
    - Model is overfitted, but at least it is better than nothing
- Gloss to Sentence (Done?)
    - Maybe need fine-tuning?
- Docker container for LLM and seq2seq model (Done, but not tested yet)
- (Optional) Website for model usage

## FAQ

1. Can the encoder and decoder be used on low-end devices via LiteRT? (formerly known as TFLite)
    - A: **No**, because LSTM and GRU are not supported in LiteRT. This is why the model is hosted as API.
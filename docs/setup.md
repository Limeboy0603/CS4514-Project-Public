# System Requirements and Setup Instructions

For real time usage, you will need a **30 FPS webcam** to use the software. If necessary, you may use OBS's virtual camera feature with lowered output FPS. You can also skip this step and use pre-recorded videos instead, which you can set in config file in `config`.

You can run this on any OS with python3 installed. Simply install all dependencies. It is highly recommended that you create a virtual environment first.

### Linux/MacOS
```sh
# Create envirionment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Windows
```sh
# Create envirionment
# Note: depending on how you installed python, your system might use the alias `python` instead of `python3`
python3 -m venv .venv
.\.venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt
```

## Configurations

A yaml config file can be found in `config/config_clip.yaml`. Simply follow the instructions in the yaml comments.

Note that Windows uses backslashes as opposed to forward slash. Thus, you should use `\\` instead of `/` for Windows. Alternatively, a config file for Linux and MacOS systems is provided at `config/config_clip_linux.yaml`

For a detailed documentation of configuration files, please check `config.md`

## Obtaining Dataset and Models

For your convenience, a copy of the dataset, the sign language recognition model and the fine-tuned model are placed on the gateway server.

```sh
scp -r $EID$@gateway.cs.cityu.edu.hk:/public/dickmwong3/CS4514/* .
```

## Building the models on your own

Alternatively, you can manually build the model on your own machine. We provide a python script that will run everything in order. *(though it would be MUCH quicker if you simply copy a bunch of directories from gateway)*

```sh
scp -r dickmwong3@gateway.cs.cityu.edu.hk:/public/dickmwong3/CS4514/* .

python3 src/!FULL.py --config $PATH_TO_CONFIG_FILE 
# By default, if you do not include --config, it will use config/config_clip.yaml for windows and config/config_clip_linux.yaml on linux/MacOS

# Here are some other flags you can use. 
# --capture: Capture glosses in dataset based on whatever you set in your config file. If you copied the dataset from the gateway server, do NOT include this arguement. Otherwise, you may use this flag to capture your own dataset.
# --extractsplit: Perform feature extraction and splitting. If you copied the dataset from the gateway server, do NOT include this arguement.
# --use: Use the model directly after building everything. The video source will depend on what you set in your config file.
```
# System Requirements and Setup Instructions

For real time usage, you will need a **30 FPS webcam** to use the software. If necessary, you may use OBS's virtual camera feature with lowered output FPS. If you feel like so, you can also skip this step and use pre-recorded videos instead, which you can set in config file. (see below)

You can run this on any OS with python3 installed. Simply install all dependencies. It is highly recommended that you create a virtual environment first.

### Linux/MacOS
```sh
# Create envirionment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Windwos
```sh
# Create envirionment
python3 -m venv .venv
.\.venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt
```

## Configurations

A yaml config file can be found in `config/config_clip.yaml`. Simply follow the instructions in the yaml comments.

Note that Windows uses backslashes as opposed to forward slash. Thus, you should use `\\` instead of `/` for Windows. Alternatively, a config file for Linux and MacOS systems is provided at `config/config_clip_linux.yaml`

For a detailed documentation of configuration files, please check `config.md`

## Obtaining Dataset

For privacy concerns (since this repository is public), the dataset is NOT included in this repository.

A copy of the dataset is placed on the gateway server in the following location: `/public/dickmwong3/CS4514/dataset`

A directory named `dataset` can be found in the above location. `scp` the entire directory to the project's working directory.

```sh
scp -r $EID@gateway.cs.cityu.edu.hk:/public/dickmwong3/CS4514/dataset .
```

## Obtaining the Model

A copy of the final SLR model is placed on the gateway server in the following location: `/public/dickmwong3/CS4514/models`

A copy of the final language model is also placed on the gateway server in the following location: `/public/dickmwong3/CS5414/llm`

Simply copy both model directories into your working directory.

```sh
scp -r $EID@gateway.cs.cityu.edu.hk:/public/dickmwong3/CS4514/models .
scp -r $EID@gateway.cs.cityu.edu.hk:/public/dickmwong3/CS4514/llm .
```

Alternatively, you can manually build the model on your own machine. We provide a python script that will run everything in order. *(though it would be MUCH quicker if you simply copy a file from gateway)*

```sh
python3 src/!FULL.py --config $PATH_TO_CONFIG_FILE 
# By default, if you do not include --config, it will use config/config_clip.yaml

# Here are some other flags you can use. 
# --capture: Capture glosses in dataset based on whatever you set in your config file. If you copied the dataset from the gateway server, do NOT include this arguement.
# --extractsplit: Perform feature extraction and splitting. If you copied the dataset from the gateway server, do NOT include this arguement.
# --use: Use the model directly after building everything. The video source will depend on what you set in your config file.
```
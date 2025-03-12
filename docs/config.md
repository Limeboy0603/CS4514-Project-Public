# Configurations

This documentation details the instructions of configurating the software. There are 2 configration files provided:

- Windows: `config/config_clip.yaml`
- Linux/MacOS: `config/config_clip_linux.yaml`

The main difference of these 2 config files lies in the paths, as Windows uses `\` instead of `/`

Examples of each config are written in code blocks.

## `capture`

Contains settings regarding the video input, which will be used by OpenCV's VideoCapture. 

### `capture.source`

Sets the source of the video input. This can either be:

- The ID of your camera. Check your connected cameras using the command below:
    - Windows: `Get-CimInstance Win32_PnPEntity | ? { $_.service -eq "usbvideo" } | Select-Object -Property PNPDeviceID, Name`
    - Linux: `ls /dev/video*`
    - MacOS: `system_profiler SPCameraDataType`
- A path to a video

```yaml
  # Using camera
  source: 0

  # Using video
  source: "dataset\\minified_3_fixed\\full_test\\final.avi"
```

### `capture.resolution`

Sets the resolution of the video input. It should be in the form of `{width}x{height}`

```yaml
  resolution: 1920x1080
```

## `paths`

Contains settings to directories and files necessary for the system. 

By default, if the directory does not exist, it will be automatically be created.

### `paths.model`

Sets the path of the model output after training and the model to use for the application.

```yaml
  model: "models/final.keras"
```

### `paths.keypoints`

Sets the output directory of the extracted features.

```yaml
  keypoints: "dataset\\minified_3_fixed\\keypoints"
```

### `paths.model_checkpoint`

Sets the output file location of model checkpoint during training.

```yaml
  model_checkpoint: "model_checkpoint/model_checkpoint.keras"
```

### `paths.split`

Sets the output directory of extracted featuers after performing train-test-val split.

```yaml
  split: "dataset\\minified_3_fixed\\split"
```

### `paths.llm`

Sets the directory name to store the language model and it's training result.

```yaml
  llm: "llm"
```

## `dictionary`

A list that defines the glosses to capture using `src/cam_capture.py`

```yaml
dictionary:
  # - "#DEMO" # the gloss for demonstration capturing program
  - "#BLANK"
  - Hello
  - Me
  - English
  - Name
  - D
  - I
  - C
  - K
```

## `sequence`

Defines the amount of data in the dataset.

### `sequence.frame`

Defines how long each clip in the dataset is in the unit of frames.

```yaml
  frame: 30
```

### `sequence.count`

Defines how many clips does each gloss label contain.

```yaml
  count: 100
```

### `llm_samples`

Contains the samples to train the BART model.

It should be a list of objects, each object containing 2 attributes
- `hksl`: Sentence in HKSL grammar
- `natural`: Natural sentence to be translated from `hksl`

```yaml
llm_samples:
  - hksl: Hello Me English Name D I C K
    natural: Hello, my English name is Dick.
  - hksl: Hello D I C K
    natural: Hello, Dick.
  - hksl: Hello Me English Name D E R E K
    natural: Hello, my English name is Derek.
  - hksl: Hello Me English Name D I C K S O N
    natural: Hello, my English name is Dickson.
  - hksl: Hello Me English Name R I C K
    natural: Hello, my English name is Rick.
```
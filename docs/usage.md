# Usage

## Capturing Dataset

**Note: This assumes that you did NOT copy the dataset from gateway server and are building your own. Disregard this part if you copied the dataset.**

You should change the following settings in your config file first.
- Change `capture.source` to a camera instead of a video
- Make sure that `capture.resolution` is correct as well
- Set the list of glosses you want to capture in `dictionary`
- Set the number of frames per clip and number of clips you want to capture in `sequence`

You should also make sure that the glosses you intend to capture should also appear in sentence sammples in `llm_samples`

Once you have set up everything, simply run the following script to start capturing your own dataset.

```sh
python3 src/cam_capture.py
```

## Using the Model
Simply run

```sh
python3 src/app.py
```

### Button usages

- `Start`: Starts the camera for isolated sign language classification.
- `Stop`: Stops the camera for isolated sign language classification.
- `Predict Natural Sentence`: Predicts a sentence based on the sequence of recorded glosses.
- `Clear`: Clear all cached sequence of gloses and predicted sentence.
- `Kill Thread`: Kills the thread for isolated sign language classification. 
- `Restart`: Restarts the thread. This will kill the thread first before restarting.
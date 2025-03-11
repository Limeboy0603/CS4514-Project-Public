# Usage

Simply run

```sh
python3 src/app.py
```

## Button usages

- `Start`: Starts the camera for isolated sign language classification.
- `Stop`: Stops the camera for isolated sign language classification.
- `Predict Natural Sentence`: Predicts a sentence based on the sequence of recorded glosses.
- `Clear`: Clear all cached sequence of gloses and predicted sentence.
- `Kill`: Kills the thread for isolated sign language classification. 
- `Restart`: Restarts the thread. This will kill the thread first before restarting.
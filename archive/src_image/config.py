import yaml
from typing import Union
import os

class Config_Capture:
    def __init__(self, source: Union[str, int], resolution: str):
        assert "x" in resolution
        assert resolution.split("x")[0].isdigit()
        assert resolution.split("x")[1].isdigit()
        
        self.source = source
        self.resolution_width = int(resolution.split("x")[0])
        self.resolution_height = int(resolution.split("x")[1])

class Config_Paths:
    def __init__(self, keypoints: str, model: str, model_checkpoint: str, dataset: str):
        os.makedirs(keypoints, exist_ok=True)
        os.makedirs(os.path.dirname(model), exist_ok=True)
        os.makedirs(os.path.dirname(model_checkpoint), exist_ok=True)
        os.makedirs(dataset, exist_ok=True)
        
        self.keypoints = keypoints
        self.model = model
        self.model_checkpoint = model_checkpoint
        self.dataset = dataset

class Config:
    def __init__(self, capture: Config_Capture, paths: Config_Paths, dictionary: list[str]):
        self.capture = capture
        self.paths = paths
        self.dictionary = dictionary

def config_parser(path: str) -> Config:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        capture = Config_Capture(
            config["capture"]["source"],
            config["capture"]["resolution"]
        )
        paths = Config_Paths(
            config["paths"]["keypoints"],
            config["paths"]["model"],
            config["paths"]["model_checkpoint"],
            config["paths"]["dataset"]
        )
        dictionary = config["dictionary"]
        return Config(capture, paths, dictionary)
        

if __name__ == "__main__":
    config_parser("config/config_image.yaml")
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
    def __init__(self, keypoints: str, model: str, model_checkpoint: str, dataset: str, split: str, llm: str):
        os.makedirs(keypoints, exist_ok=True)
        os.makedirs(os.path.dirname(model), exist_ok=True)
        os.makedirs(os.path.dirname(model_checkpoint), exist_ok=True)
        os.makedirs(dataset, exist_ok=True)
        os.makedirs(split, exist_ok=True)
        os.makedirs(llm, exist_ok=True)
        
        self.keypoints = keypoints
        self.model = model
        self.model_checkpoint = model_checkpoint
        self.dataset = dataset
        self.split = split
        self.llm = llm

        self.split_train = os.path.join(split, "train")
        self.split_val = os.path.join(split, "val")
        self.split_test = os.path.join(split, "test")

        self.llm_model = os.path.join(llm, "model")
        self.llm_results = os.path.join(llm, "results")

class Config_Sequence:
    def __init__(self, frame: int, count: int):
        assert frame > 0
        assert count > 0

        self.frame = frame
        self.count = count

class Config_LLM_Item:
    def __init__(self, hksl: str, natural: str):
        self.hksl = hksl
        self.natural = natural

class Config_LLM:
    def __init__(self, items: list[dict[str, str]]):
        self.llm_samples = []
        for item in items:
            self.llm_samples.append(Config_LLM_Item(item["hksl"], item["natural"]))

        self.hksl = [item.hksl for item in self.llm_samples]
        self.natural = [item.natural for item in self.llm_samples]

        assert len(self.hksl) == len(self.natural)

class Config:
    def __init__(self, capture: Config_Capture, paths: Config_Paths, dictionary: list[str], sequence: Config_Sequence, llm: Config_LLM):
        self.capture = capture
        self.paths = paths
        self.dictionary = dictionary
        self.sequence = sequence
        self.llm = llm

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
            config["paths"]["dataset"],
            config["paths"]["split"],
            config["paths"]["llm"],
        )
        dictionary = config["dictionary"]
        sequence = Config_Sequence(
            config["sequence"]["frame"],
            config["sequence"]["count"]
        )
        llm = Config_LLM(
            config["llm_samples"]
        )
        return Config(capture, paths, dictionary, sequence, llm)

if __name__ == "__main__":
    config_parser("config/config_clip.yaml" if os.name == "nt" else "config/config_clip_linux.yaml")
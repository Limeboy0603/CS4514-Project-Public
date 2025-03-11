import os

from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from config import Config, config_parser

def slt_sentence_predict(sentence: str, config: Config) -> str:
    llm_model = BartForConditionalGeneration.from_pretrained(config.paths.llm_model)
    tokenizer = BartTokenizer.from_pretrained(config.paths.llm_model)
    print("LLM Model loaded")
    inputs = tokenizer(sentence, return_tensors="pt", max_length=128, truncation=True, padding='max_length')
    # print("Inputs", inputs)
    with torch.no_grad():
        pred = llm_model.generate(**inputs)
        # print("Prediction", pred)
    result = tokenizer.decode(pred[0], skip_special_tokens=True)
    # print("Result", result)
    return result

if __name__ == '__main__':
    sentence = "Hello Me English Name D I C K"
    print(slt_sentence_predict(sentence, config_parser("config/config_clip.yaml" if os.name == "nt" else "config/config_clip_linux.yaml")))
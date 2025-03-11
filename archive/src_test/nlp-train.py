from transformers import LlamaForCausalLM, PreTrainedTokenizerFast, TrainingArguments, Trainer
from datasets import Dataset
from torch.optim import AdamW
import torch
import pandas as pd

split_file = pd.read_csv("dataset/tvb-hksl-news/split/train.csv", delimiter="|")
split_file_X = split_file["glosses"].astype(str).tolist()
split_file_Y = split_file["words"].astype(str).tolist()
# print(split_file_X)
# print(split_file_Y)

# Load pre-trained model and tokenizer
model_name = "lordjia/Llama-3-Cantonese-8B-Instruct"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)
model.to("cuda")

dataset = Dataset.from_dict({"X": split_file_X, "Y": split_file_Y})

training_args = TrainingArguments(
    output_dir="model/models_nlp",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_dir="model/logs_nlp",
    logging_steps=10,
    overwrite_output_dir=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=None,
    train_dataset=dataset
)

trainer.train()

# Save model and tokenizer
model.save_pretrained("model/models_nlp")
tokenizer.save_pretrained("model/models_nlp")

# Test the model
input_sequence = "今 溫度 一 九 濕 百分比 七 四"

# Tokenize input
input_ids = tokenizer.encode(input_sequence, return_tensors='pt').to("cuda")

# Generate sentence
output = model.generate(input_ids, max_length=200, num_return_sequences=1)

# Decode output
generated_sentence = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_sentence)
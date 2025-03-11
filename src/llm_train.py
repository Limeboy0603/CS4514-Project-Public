import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from datasets import Dataset, DatasetDict
import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from config import config_parser

def main(config_path: str):
    config = config_parser(config_path)

    df = pd.DataFrame({
        "gloss": config.llm.hksl,
        "text": config.llm.natural,
    })

    train_dataset = Dataset.from_pandas(df)
    dataset = DatasetDict({
        "train": train_dataset,
    })
    # print(dataset)

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        inputs = examples['gloss']
        targets = examples['text']
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length')
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    # Tokenize dataset
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Model
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    model.config.pad_token_id = tokenizer.eos_token_id

    training_args = TrainingArguments(
        output_dir=config.paths.llm_results,
        evaluation_strategy='epoch',
        learning_rate=2e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=32,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['train'],
    )

    trainer.train()

    # Save the model and the tokenizer
    # model.save_pretrained("./llm_model")
    # tokenizer.save_pretrained("./llm_model")
    model.save_pretrained(config.paths.llm_model)
    tokenizer.save_pretrained(config.paths.llm_model)

if __name__ == "__main__":
    main("config/config_clip.yaml" if os.name == "nt" else "config/config_clip_linux.yaml")
    from llm_util import slt_sentence_predict
    sentence = "Hello Me English Name D I C K"
    print(slt_sentence_predict(sentence, config_parser("config/config_clip.yaml" if os.name == "nt" else "config/config_clip_linux.yaml")))
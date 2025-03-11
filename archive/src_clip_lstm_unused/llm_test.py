import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from datasets import load_dataset
dataset = load_dataset("achrafothman/aslg_pc12")
print(dataset)
"""
DatasetDict({
    train: Dataset({
        features: ['gloss', 'text'],
        num_rows: 87710
    })
})

Task: Sequence-to-sequence translation from `gloss` to `text
"""

from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

def preprocess_function(examples):
    inputs = examples['gloss']
    targets = examples['text']
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Tokenize dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Model
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['train'],
)

# Train the model
trainer.train()
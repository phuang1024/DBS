from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

dataset = load_dataset("glue", "stsb")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["sentence1"],
        batch["sentence2"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.remove_columns(["sentence1", "sentence2", "idx"])
dataset.set_format("torch")

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=1,
)

args = TrainingArguments(
    output_dir="./results/bert_stsb",

    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,

    learning_rate=2e-5,

    report_to=["tensorboard"],
    logging_strategy="steps",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=1000,
)

trainer = Trainer(
    model=model,
    args=args,

    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
)

trainer.train()

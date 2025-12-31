from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

dataset = load_dataset("imdb")

#dataset["train"] = dataset["train"].shuffle(seed=42).select(range(1000))
#dataset["test"] = dataset["test"].shuffle(seed=42).select(range(500))

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.remove_columns(["text"])
dataset.set_format("torch")

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
)

args = TrainingArguments(
    output_dir="./bert_ckpts",

    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,

    report_to=["tensorboard"],
    logging_strategy="steps",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=200,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)

trainer.train()

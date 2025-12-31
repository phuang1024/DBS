"""
Create and load huggingface model and datasets.
"""

from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
)


def create_dataset():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    def tokenize(batch):
        return tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    dataset = load_dataset("glue", "stsb")

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.remove_columns(["sentence1", "sentence2", "idx"])
    dataset.set_format("torch")

    return dataset, tokenizer


def create_model():
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=1,
    )
    return model

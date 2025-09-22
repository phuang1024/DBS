"""
Load models and datasets for testing.
Used by test_llm.py
"""

from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)
from datasets import load_dataset

TRAIN_LEN = 5000
EVAL_LEN = 500


def load_bert():
    dataset = load_dataset("glue", "sst2")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=64)

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_dataset = dataset["train"].shuffle(seed=42).select(range(TRAIN_LEN))
    eval_dataset = dataset["validation"].shuffle(seed=42).select(range(EVAL_LEN))

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    return model, train_dataset, eval_dataset


def load_gpt2():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token

    def tok(batch):
        tokens = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    dataset = dataset.map(tok, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    train_dataset = dataset["train"].select(range(TRAIN_LEN))
    eval_dataset = dataset["validation"].select(range(EVAL_LEN))

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    return model, train_dataset, eval_dataset


def load_model(type):
    if type == "bert":
        return load_bert()
    elif type == "gpt2":
        return load_gpt2()
    else:
        raise ValueError(f"Unknown model type: {type}")
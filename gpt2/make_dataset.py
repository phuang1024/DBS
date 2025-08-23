"""
Make dataset from text files.
"""

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

tok = PreTrainedTokenizerFast.from_pretrained("./_tokenizer")

def tokenize(batch):
    return tok(
        batch["text"],
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

def group_texts(examples, block_size=512):
    # Concatenate then split into fixed blocks
    ids = sum(examples["input_ids"], [])
    total_length = (len(ids) // block_size) * block_size
    ids = ids[:total_length]
    blocks = [ids[i:i+block_size] for i in range(0, total_length, block_size)]
    return {"input_ids": blocks, "labels": blocks.copy(), "attention_mask": [[1]*block_size]*len(blocks)}

raw = load_dataset("text", data_files={"train":"data/train.txt","validation":"data/valid.txt"})
tokenized = raw.map(tokenize, batched=True, remove_columns=["text"])
lm_datasets = tokenized.map(group_texts, batched=True)

lm_datasets.save_to_disk("./_dataset")


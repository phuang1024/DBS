"""
Run single process:
python test_llm.py
Run DDP multi process:
torchrun --nproc_per_node=2 test_llm.py
"""

import torch
import torch.distributed as dist
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

from ddp_hook import ddp_eg_coding, EGHookState, _vanilla

dist.init_process_group("gloo")

# 1. Load a small dataset (SST2 subset)
dataset = load_dataset("glue", "sst2")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=64)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))   # tiny subset
eval_dataset = dataset["validation"].shuffle(seed=42).select(range(500))

# 2. Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 3. Training arguments
training_args = TrainingArguments(
    output_dir="./bert_out",
    eval_strategy="epoch",
    save_strategy="no",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    logging_steps=50,
    fp16=False,
    report_to="none",   # disable wandb/mlflow/etc.
)

# 4. Trainer
hook_state = EGHookState()

class EGTrainer(Trainer):
    def _wrap_model(self, model, *args, **kwargs):
        model = super()._wrap_model(model, *args, **kwargs)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            # Register DDP hook
            model.register_comm_hook(state=hook_state, hook=_vanilla)
        return model

trainer = EGTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 5. Train
trainer.train()

# 6. Print hook results
if dist.is_initialized() and dist.get_rank() == 0:
    print(f"DDP Hook calls: {hook_state.calls}")
    print(f"Total params transferred: {hook_state.params}")
    print(f"Total bytes transferred: {hook_state.bytes}")

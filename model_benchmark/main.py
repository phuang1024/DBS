"""
Main training script for model benchmarking.

Supports various customizations:
- Device and DDP backend.
- Quantization.
- Gradient compression algorithms.
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from transformers import (
    TrainingArguments,
    Trainer,
)

from model import create_dataset, create_model

# Toggle distributed training.
DO_DIST = False


def train(rank, world_size):
    if DO_DIST:
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)

        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )

    dataset, tokenizer = create_dataset()
    model = create_model()

    args = TrainingArguments(
        output_dir="./results",

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,

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

    if DO_DIST:
        dist.destroy_process_group()


def spawn_dist():
    world_size = 2

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "13579"

    mp.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    if DO_DIST:
        spawn_dist()
    else:
        train(None, None)

"""
Main training script for model benchmarking.

Supports various customizations:
- Device and DDP backend.
- Quantization.
- Gradient compression algorithms.
"""

import argparse

from transformers import (
    TrainingArguments,
    Trainer,
)

from model import create_dataset, create_model


def main():
    dataset, tokenizer = create_dataset()
    model = create_model()

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


if __name__ == "__main__":
    main()

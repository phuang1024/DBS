import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

from ddp_hook import ddp_eg_coding, EGHookState, _noop
from test_llm_models import load_model

WORLD_SIZE = 2


hook_state = EGHookState()

class EGTrainer(Trainer):
    def _wrap_model(self, model, *args, **kwargs):
        model = super()._wrap_model(model, *args, **kwargs)

        # Register DDP hook
        if not isinstance(model, DDP):
            model = DDP(model)
        model.register_comm_hook(state=hook_state, hook=ddp_eg_coding)

        return model


def train(rank):
    # Set rank to "0" because I only have one GPU.
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = str(WORLD_SIZE)

    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)

    model, train_dataset, eval_dataset = load_model("gpt2")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    print("Training.")
    training_args = TrainingArguments(
        output_dir="./bert_out",
        eval_strategy="epoch",
        save_strategy="no",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        logging_steps=50,
        fp16=False,
        report_to="none",
        ddp_backend="gloo",
    )
    trainer = EGTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    time_start = time.time()
    trainer.train()
    elapse = time.time() - time_start
    if rank == 0:
        print(f"Training time: {elapse:.2f} seconds")

    if rank == 0:
        state = hook_state
        print(f"DDP Hook calls: {state.calls}")
        print(f"Total params transferred: {state.params}")
        print(f"Total bytes transferred: {state.bytes}")
        #print(f"Profiling: {state.profiling[:7]}")

    print("Evaluating")
    eval_results = trainer.evaluate()
    if rank == 0:
        print(f"Eval results: {eval_results}")

    print(f"Rank {rank} finished.")


def main():
    mp.spawn(
        train,
        args=(),
        nprocs=WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()

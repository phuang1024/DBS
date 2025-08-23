"""
Train model.
"""

DO_BACKSLASH = False

from datasets import load_from_disk
from transformers import (
    GPT2Config, GPT2LMHeadModel,
    PreTrainedTokenizerFast, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)

from backslash import backslash

tok = PreTrainedTokenizerFast.from_pretrained("./_tokenizer")
ds = load_from_disk("./_dataset")

config = GPT2Config.from_pretrained("./_model.json")
model = GPT2LMHeadModel(config)

collator = DataCollatorForLanguageModeling(tok, mlm=False)


class BackslashCallback(TrainerCallback):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        if model is not None:
            self.fn(model)
        return control


# Train phase one: Yes backslash
if DO_BACKSLASH:
    print("Training phase one: Yes backslash")

    args = TrainingArguments(
        output_dir="ckpts/tiny-gpt",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,   # effective batch 16
        num_train_epochs=3,
        learning_rate=3e-4,
        warmup_ratio=0.03,
        weight_decay=0.1,
        logging_steps=50,
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        fp16=True,                       # set False if CPU
        bf16=False,                      # True if your GPU supports bfloat16
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        tokenizer=tok,
        callbacks=[BackslashCallback(backslash)],
    )


# Train phase two: No backslash
print("Training phase two: No backslash")

args = TrainingArguments(
    output_dir="ckpts/tiny-gpt",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,   # effective batch 16
    num_train_epochs=3,
    learning_rate=3e-4,
    warmup_ratio=0.03,
    weight_decay=0.1,
    logging_steps=50,
    eval_steps=200,
    save_steps=200,
    save_total_limit=2,
    fp16=True,                       # set False if CPU
    bf16=False,                      # True if your GPU supports bfloat16
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=collator,
    tokenizer=tok
)


trainer.train()
trainer.save_model("tiny-gpt")
tok.save_pretrained("tiny-gpt")

import torch

all_params = torch.cat([p.flatten().detach().cpu() for p in model.parameters()])
path = ("yes" if DO_BACKSLASH else "no") + "_backslash_params.pt"
print("Saving parameters to", path)
torch.save(all_params, path)

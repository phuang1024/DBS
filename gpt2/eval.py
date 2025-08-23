"""
Eval and generate text.
"""

import math, evaluate, torch
from datasets import load_from_disk
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

tok = PreTrainedTokenizerFast.from_pretrained("tiny-gpt")
model = GPT2LMHeadModel.from_pretrained("tiny-gpt")
model.eval()

# Perplexity on valid set
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
ds = load_from_disk("./_dataset")["validation"]
collator = DataCollatorForLanguageModeling(tok, mlm=False)
loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collator)

loss_sum = 0
count = 0
with torch.no_grad():
    for batch in loader:
        batch = {k:v.to(model.device) for k,v in batch.items()}
        out = model(**batch)
        loss_sum += out.loss.item() * batch["input_ids"].size(0)
        count += batch["input_ids"].size(0)
ppl = math.exp(loss_sum / count)
print("Perplexity:", round(ppl, 2))

# Text generation
prompt = "Once upon a midnight"
inputs = tok(prompt, return_tensors="pt").to(model.device)
gen = model.generate(
    **inputs,
    max_new_tokens=80,
    do_sample=True,
    top_p=0.9,
    temperature=0.8,
    eos_token_id=tok.eos_token_id
)
print(tok.decode(gen[0], skip_special_tokens=True))


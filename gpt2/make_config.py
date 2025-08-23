"""
Train tokenizer and save to disk.
Make model config and save to disk.
"""

# Train tokenizer.
print("Training tokenizer")
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tok = Tokenizer(BPE(unk_token="[UNK]"))
tok.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=8000,
    min_frequency=2,
    special_tokens=["[PAD]","[UNK]","[BOS]","[EOS]"]
)
tok.train(files=["data/train.txt", "data/valid.txt"], trainer=trainer)
tok.save("_tokenizer.json")

# Save tokenizer in transformers format.
print("Saving tokenizer in transformers format")
from transformers import PreTrainedTokenizerFast

tok = PreTrainedTokenizerFast(
    tokenizer_file="_tokenizer.json",
    bos_token="[BOS]",
    eos_token="[EOS]",
    unk_token="[UNK]",
    pad_token="[PAD]"
)
tok.save_pretrained("./_tokenizer")


# Make model config.
print("Making model config")
from transformers import GPT2Config
from tokenizers import Tokenizer

t = Tokenizer.from_file("_tokenizer.json")
vocab_size = t.get_vocab_size()

config = GPT2Config(
    vocab_size=vocab_size,
    n_positions=512,    # context length
    n_ctx=512,
    n_embd=256,         # hidden size
    n_layer=4,          # transformer blocks
    n_head=4,           # attention heads
    bos_token_id=t.token_to_id("[BOS]"),
    eos_token_id=t.token_to_id("[EOS]")
)
with open("_model.json","w") as f:
    f.write(config.to_json_string())

# DBS

Distributed Backslash.

## Testing instructions

Around 4 hours per run.

Tensorboard and other logs are saved to runs/*.

```bash
pip install -r requirements.txt

cd dbs
python test_llm.py bert
python test_llm.py gpt2
```

## Files

- `dbs/`: Main source code.
- `ddp/`: Experiments on gradient compression with Torch DDP.
- `eg/`: Experiments on Exp-Golomb coding.
- `gg_prior`: Source from Backslash paper.
- `gpt2`: Experiments on GG constrained training of GPT-2.

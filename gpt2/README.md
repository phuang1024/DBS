Train GPT2 with Backslash using Huggingface.

Run in this order:

- make_config.py: Make tokenizer and model config.
- make_dataset.py: Make dataset from text file.
- train.py: Train model.
    Set the global var `DO_BACKSLASH` to True and False. Run twice.
- eval.py: Evaluate and generate text.
- plot_params.py: Plot model parameter distribution.

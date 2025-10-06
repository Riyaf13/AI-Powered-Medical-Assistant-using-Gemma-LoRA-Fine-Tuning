# data_loader.py
"""
Data loader and preprocessing for AlpaCare-MedInstruct-52k
- Downloads from Hugging Face
- Cleans basic issues
- Creates train/val/test split (90/5/5 by default)
- Tokenizes for a causal LM tokenizer
- Exposes a function get_datasets(tokenizer, max_length, split_seed)
"""

from datasets import load_dataset
import re
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import random

def basic_clean(text: str) -> str:
    # basic cleaning: normalize whitespace, remove non-printables
    if text is None:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    # strip out control characters
    text = ''.join(ch for ch in text if ch.isprintable())
    return text

def format_example(example: dict) -> dict:
    """
    Adapt to dataset format. The AlpaCare-MedInstruct dataset contains fields
    depending on its schema; common fields are 'instruction', 'input', 'output' or 'response'.
    We'll create a single 'text' field that contains instruction + input + output prompt style.
    """
    # try several common keys
    instr = example.get("instruction") or example.get("prompt") or ""
    inp = example.get("input") or example.get("context") or ""
    out = example.get("output") or example.get("response") or example.get("answer") or ""
    instr = basic_clean(instr)
    inp = basic_clean(inp)
    out = basic_clean(out)
    # Build a simple prompt-response formatting (non-diagnostic template)
    # Note: Do NOT include any diagnostic/prescription in the prompt template — the dataset may contain them.
    text = f"### Instruction:\n{instr}\n"
    if inp:
        text += f"\n### Input:\n{inp}\n"
    text += f"\n### Response:\n{out}\n"
    return {"text": text}

def prepare_and_split(dataset_name: str = "lavita/AlpaCare-MedInstruct-52k",
                      text_column: str = None,
                      seed: int = 42,
                      train_frac: float = 0.90,
                      val_frac: float = 0.05,
                      test_frac: float = 0.05,
                      max_examples: int = None) -> dict:
    """
    Downloads dataset, normalizes, and splits into train/val/test.
    Returns a dict with 'train', 'validation', 'test' datasets (HF Dataset objects).
    """
    raw = load_dataset(dataset_name)
    # If dataset has multiple splits, merge them
    if "train" in raw and len(raw) == 1:
        ds = raw["train"]
    else:
        # pick the first split or concatenate splits
        try:
            ds = raw[list(raw.keys())[0]]
        except:
            ds = raw
    # Map to text field
    # If there's a single text column we can use it
    possible_text_cols = ["text", "instruction", "prompt", "response", "output"]
    if text_column is None:
        for c in possible_text_cols:
            if c in ds.column_names:
                text_column = c
                break

    if text_column:
        # keep text column directly
        def mapper(ex):
            txt = basic_clean(ex.get(text_column, ""))
            return {"text": txt}
        ds = ds.map(mapper, remove_columns=[c for c in ds.column_names if c != text_column])
    else:
        # try to format using common fields
        ds = ds.map(format_example, remove_columns=ds.column_names)

    # Optionally reduce the dataset for demo (Colab GPU limits)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    # Shuffle and split
    ds = ds.shuffle(seed=seed)
    n = len(ds)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    train_ds = ds.select(range(0, train_end))
    val_ds = ds.select(range(train_end, val_end))
    test_ds = ds.select(range(val_end, n))

    return {"train": train_ds, "validation": val_ds, "test": test_ds}

def tokenize_datasets(dsets: dict, tokenizer: PreTrainedTokenizerBase, max_length: int = 512) -> dict:
    """
    Tokenize datasets for causal LM. Pads/truncates to max_length.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    def tokenize_fn(examples):
        # tokenizer expects text inputs
        out = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)
        # For causal LM, labels = input_ids
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = {}
    for split, ds in dsets.items():
        tokenized[split] = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
    return tokenized

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tkn = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    if tkn.pad_token is None:
        tkn.pad_token = tkn.eos_token   # <-- add this line
    d = prepare_and_split(max_examples=1000)
    tok = tokenize_datasets(d, tkn, max_length=512)
    print({k: len(v) for k,v in tok.items()})

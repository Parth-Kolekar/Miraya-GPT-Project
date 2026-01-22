# Save this as: data/dolly_prepare.py
import os
import numpy as np
import tiktoken
from datasets import load_dataset

# Configuration
out_dir = 'data/dolly'
os.makedirs(out_dir, exist_ok=True)

# 1. Load the dataset (Databricks Dolly 15k)
print("Downloading dataset...")
dataset = load_dataset("databricks/databricks-dolly-15k", split='train')

# 2. Split into Train (90%) and Validation (10%)
split_dataset = dataset.train_test_split(test_size=0.1, seed=1337, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')

# 3. Tokenizer (GPT-2 standard)
enc = tiktoken.get_encoding("gpt2")

def process(example):
    # Format: "User: <instruction> \nAssistant: <response> <|endoftext|>"
    # We add <|endoftext|> so the model knows when to stop talking.
    text = f"User: {example['instruction']}\nAssistant: {example['response']}<|endoftext|>"
    ids = enc.encode_ordinary(text)
    return {'ids': ids, 'len': len(ids)}

# 4. Tokenize
print("Tokenizing data...")
tokenized = split_dataset.map(
    process,
    remove_columns=['instruction', 'context', 'response', 'category'],
    desc="Processing",
)

# 5. Save to .bin files
for split, dset in tokenized.items():
    print(f"Writing {split}.bin...")
    filename = os.path.join(out_dir, f'{split}.bin')
    dtype = np.uint16 # GPT-2 vocabulary fits in uint16
    arr_len = np.sum(dset['len'], dtype=np.int64)
    
    # Create a memory-mapped array to write to disk
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    
    idx = 0
    for example in dset:
        arr[idx : idx + example['len']] = example['ids']
        idx += example['len']
    arr.flush()

print("Done! Data is ready in data/dolly/")
# Save this as: config/finetune_dolly.py
import time

# I/O
out_dir = 'out-dolly'
eval_interval = 50
eval_iters = 40
wandb_log = False # Set to True if you want to log to wandb
always_save_checkpoint = False

# Data
dataset = 'dolly'
init_from = 'gpt2' # TRANSFER LEARNING: Start with OpenAI's pre-trained weights

# Optimization for RTX 4060 (8GB VRAM)
# We use gradient accumulation to simulate a larger batch size
batch_size = 4        # Micro-batch size (fits in VRAM)
block_size = 1024     # Context length
gradient_accumulation_steps = 32 # 4 * 32 = 128 effective batch size

max_iters = 500       # 500 steps is usually enough for a small dataset
learning_rate = 3e-5  # Low learning rate because we are fine-tuning
decay_lr = False
warmup_iters = 50

# Hardware Settings
device = 'cuda'       # Force GPU
dtype = 'bfloat16'    # RTX 40-series supports this (faster & stable)
compile = True        # PyTorch 2.0 compilation (huge speedup)

# System
# If compile=True crashes on Windows, set it to False.
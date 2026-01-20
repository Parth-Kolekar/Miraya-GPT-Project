# config/train_tinystories.py
out_dir = 'out-tinystories'
eval_interval = 100
eval_iters = 5
log_interval = 1

dataset = ''
gradient_accumulation_steps = 1
batch_size = 8
block_size = 128

# Model structure
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

# ADD THIS LINE TO FIX THE CRASH
warmup_iters = 100 

device = 'cpu'
compile = False
dtype = 'float32'
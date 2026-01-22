# Save this as: chat.py
import os
import torch
import tiktoken
from model import GPTConfig, GPT

# --- Configuration ---
out_dir = 'out-dolly' # Where your training results are
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# For Mac users:
if torch.backends.mps.is_available():
    device = 'mps'

print(f"Loading model on {device}...")

# --- Load Model ---
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']

# Clean up state dict keys if needed
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.eval()
model.to(device)

if device == 'cuda':
    model = torch.compile(model) # Optional optimization

# --- Tokenizer ---
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# --- Chat Loop ---
print("\n" + "="*30)
print("🤖 GPT-2 Instruct is Ready!")
print("Type 'quit' to exit.")
print("="*30 + "\n")

while True:
    user_input = input("\nUser: ")
    if user_input.lower() in ['quit', 'exit']:
        break

    # Prepare prompt with the specific format we trained on
    prompt = f"User: {user_input}\nAssistant: "
    
    # Encode and move to device
    x = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]

    print("Assistant: ", end="", flush=True)

    # Generate
    with torch.no_grad():
        # Stop generating when we hit <|endoftext|> or newlines if desired
        # Here we just generate up to 100 tokens
        y = model.generate(x, max_new_tokens=100, temperature=0.7, top_k=50)
        
        # Isolate the *new* tokens
        completion = y[0, x.size(1):].tolist()
        
        # Decode and print
        decoded_text = decode(completion)
        
        # Stop at <|endoftext|> if it was generated
        if "<|endoftext|>" in decoded_text:
            decoded_text = decoded_text.split("<|endoftext|>")[0]
            
        print(decoded_text)
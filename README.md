# Create Venv First

python -m venv venv

venv\Scripts\Activate.ps1

Installation:
pip install torch numpy transformers datasets tiktoken wandb tqdm

(Optional but recommended) Save dependencies
pip freeze > requirements.txt

To recreate later:
pip install -r requirements.txt

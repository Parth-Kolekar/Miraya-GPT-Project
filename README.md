# Create Venv First

python -m venv venv

venv\Scripts\Activate.ps1

Installation:
pip install torch numpy transformers datasets tiktoken wandb tqdm

(Optional but recommended) Save dependencies
pip freeze > requirements.txt

To recreate later:
pip install -r requirements.txt

📄 **Editable Google Doc:**  
[Click here to view](https://docs.google.com/document/d/1d7mQSfzNqvVleFlIhcW-l6Ubj0AbG6K1Ig4dgEmZNDc/edit?tab=t.0)



## CPU Training (Current Run on Normal Laptop)

**Task:** Training a *"Tiny"* model (approx. **2M parameters**) from scratch.

**Estimated Time:** **1 to 3 hours**

**Why:**  
Even though the model is small, CPUs are not designed for the heavy matrix math required for training. It will finish, but it takes time.

---

## GPU Training (RTX 4060 Laptop)

**Task:** Fine-tuning **GPT-2 (124M parameters)**

**Estimated Time:** **10 to 20 minutes**

**Why:**  
The RTX 4060 is specialized for this. Even though GPT-2 is ~100× larger than your tiny CPU model, the GPU is so fast it will finish in a fraction of the time. This results in a much smarter, *real* English-speaking assistant.



When we say `init_from = 'gpt2'`, we are not downloading a **"black box"** program from OpenAI.  
We are using **your code (`model.py`)** and simply filling it with **"educated" numbers** instead of random ones.

Below is a breakdown of **why we do this** and **how you can choose the "Hard Mode"** (training from scratch) if you prefer.

---

## The "Brain" Analogy

### 🧠 Training from Scratch (What You Did on CPU)

- You create your GPT. Its brain is **empty** (random noise).
- You must teach it **English, Grammar, Logic, and Storytelling** all at once.
- **Result:**  
  It takes **weeks or months of GPU time** to achieve good English.

---

### 🚀 Fine-Tuning (Recommended)

- You create your GPT (using your `model.py` code).
- You copy the **"brain patterns" (weights)** from **GPT-2**, which already understands English and grammar.
- You then train it on your **Dolly dataset** to teach it how to be an **Assistant**.
- **Result:**  
  You get a **smart chatbot in ~10 minutes**.

## HERE WE ARE USING SECOND METHOD


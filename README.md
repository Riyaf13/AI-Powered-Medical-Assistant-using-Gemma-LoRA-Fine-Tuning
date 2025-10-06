# 🩺 AI-Powered Medical Assistant using Gemma & LoRA Fine-Tuning

## 🧠 Project Overview
This project fine-tunes Google’s **Gemma 2B** model using **LoRA (Low-Rank Adaptation)** to create an **AI-powered medical assistant** capable of providing educational responses to basic health-related queries.  
The model can explain diseases, symptoms, and preventive measures in **simple language** — without providing direct diagnoses or prescriptions.  

The assistant is lightweight, privacy-safe, and optimized for **fast inference** on consumer hardware (Google Colab, mid-tier GPUs, or local systems).  

---

## 🏗️ Architecture & Approach

### 🔹 Model Architecture
- **Base Model:** `google/gemma-2b`
- **Fine-Tuning Technique:** Parameter-Efficient Fine-Tuning using **LoRA (PEFT)**
- **Training Framework:** Hugging Face `transformers` + `peft`
- **Quantization:** 8-bit precision for memory efficiency

### 🔹 Pipeline Overview
1. **Dataset Preparation:** Medical Q&A dataset in JSONL format (instruction–response pairs).  
2. **LoRA Fine-Tuning:** Only a small subset of parameters (adapters) are trained, reducing computational cost.  
3. **Adapter Saving:** The trained LoRA adapter is saved separately for flexible model merging.  
4. **Inference:** The adapter is loaded onto the base Gemma model for generation.  
5. **Safety Layer:** A disclaimer is appended to ensure compliance with AI safety practices.

### 🔹 Architecture Diagram
```
                ┌──────────────────────────┐
                │  User Query (Question)   │
                └────────────┬─────────────┘
                             │
                             ▼
                 ┌─────────────────────────┐
                 │  Tokenizer (Gemma)      │
                 └────────────┬────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │  Base Model (Gemma 2B)   │
                │  + LoRA Adapter Layers   │
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │  Generated Explanation   │
                │  + Safety Disclaimer     │
                └──────────────────────────┘
```

---

## ⚙️ Setup & Instructions to Run

### 🧩 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/gemma-medical-assistant.git
cd gemma-medical-assistant
```

### 🧩 2. Install Dependencies
```bash
pip install -r requirements.txt
```
Or install manually:
```bash
pip install torch transformers peft datasets accelerate bitsandbytes
```

### 🧩 3. Run the Notebook
Open the provided **Colab notebook** `internship_1.ipynb` or run locally:

```bash
jupyter notebook internship_1.ipynb
```

Make sure you enable GPU runtime in Colab:
```
Runtime → Change runtime type → GPU → Save
```

---

## 🧾 Dataset Information

### 🔹 Dataset Format (JSONL)
Each line in the dataset contains an instruction and its response:
```json
{"instruction": "What are the symptoms of diabetes?", 
 "input": "", 
 "output": "Common symptoms include frequent urination, excessive thirst, fatigue, and blurred vision."}
```

### 🔹 Source
You can create your own dataset or use public health datasets such as:
- **MedDialog** (medical Q&A)
- **PubMed-QA**
- **HealthCareMagic Medical QA**

Store it as:
```
data/train.jsonl
data/validation.jsonl
```

---

## ⚡ Training Process

### 🔹 Core Steps
1. Load Gemma 2B base model.  
2. Apply LoRA adapters via PEFT for efficient fine-tuning.  
3. Train on medical instruction–response pairs.  
4. Save adapter weights (`/lora_adapter`).  
5. Merge with base model for inference.

### 🔹 Quick Code Reference
```python
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", load_in_8bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, task_type="CAUSAL_LM")
model = get_peft_model(base_model, config)
```

---

## 🧪 Inference Example

```python
def safe_generate(prompt, max_new_tokens=200, temperature=0.5, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
    txt = tokenizer.decode(out[0], skip_special_tokens=True)
    return txt[len(prompt):].strip()

prompt = "Explain symptoms of diabetes in simple terms."
print(">>", safe_generate(prompt))
```

### 🧍 Sample Output
```
>> Diabetes is a condition where your body cannot properly use or make insulin.
Common symptoms include increased thirst, frequent urination, tiredness, and blurry vision.
```

---

## 📦 Dependencies

| Library | Purpose |
|----------|----------|
| `torch` | Deep learning framework |
| `transformers` | Hugging Face model loading & training |
| `peft` | LoRA fine-tuning (parameter-efficient training) |
| `bitsandbytes` | 8-bit quantization for memory efficiency |
| `datasets` | Dataset handling |
| `accelerate` | Multi-GPU / mixed-precision training support |

---

## 📊 Expected Output

| Stage | Expected Result |
|--------|-----------------|
| Training | Gradual loss reduction over epochs |
| Adapter Saving | `lora_adapter/` folder created |
| Inference | Fluent, medically-aware text response |
| Example | “Hypertension causes elevated blood pressure, often without early symptoms.” |

---

## 🧱 Project Folder Structure
```
gemma-medical-assistant/
│
├── data/
│   ├── train.jsonl
│   └── validation.jsonl
│
├── lora_adapter/
│   ├── adapter_config.json
│   └── adapter_model.bin
│
├── internship_1.ipynb
├── requirements.txt
└── README.md
```

---

## ⚖️ Ethical & Safety Note
This AI model is intended **for educational and research purposes only**.  
It **must not** be used for diagnosis, prescription, or any medical decision-making.  
Always consult a certified medical professional for accurate guidance.

---

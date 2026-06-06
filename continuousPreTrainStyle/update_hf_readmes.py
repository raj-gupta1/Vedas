import os
from huggingface_hub import HfApi

# Load HF token from .env
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
hf_token = None
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("HF_TOKEN="):
                hf_token = line.split("=", 1)[1].strip().strip('"').strip("'")

if not hf_token:
    hf_token = os.environ.get("HF_TOKEN")

if not hf_token:
    raise ValueError("Hugging Face Token (HF_TOKEN) not found in root .env or environment variables!")

# Initialize API
api = HfApi(token=hf_token)

# 1. README.md for shinigamiRaj/IndicVedas (Merged & GGUF Model)
merged_readme = """---
license: other
license_name: public-domain
language:
- en
- sa
- hi
tags:
- Vedas
- Ayurveda
- Qwen
- Qwen2.5
- unsloth
- vllm
- Text-Generation
- Sanskrit
base_model: Qwen/Qwen2.5-14B-Instruct
pipeline_tag: text-generation
---

# 🪶 VedaGPT: Merged 16-bit & GGUF Model (IndicVedas)

VedaGPT is a domain-specialized language model fine-tuned on ancient Indian scriptures (Vedas and Upanishads) and classical Ayurvedic texts (Charaka Samhita, Sushruta Samhita, Rasa Jala Nidhi, and research papers from IRJAY). 

This repository (`shinigamiRaj/IndicVedas`) hosts the **merged 16-bit (`bfloat16`) weights** and the **`q4_k_m` quantized GGUF format weights** of the fine-tuned model.

The base model is **[Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)**, fine-tuned using Unsloth's QLoRA optimization on serverless Modal GPUs.

---

## 🏛️ Model Details
- **Developer**: shinigamiRaj
- **Base Model**: `Qwen/Qwen2.5-14B-Instruct`
- **Architecture**: Causal Language Modeling
- **Max Sequence Length**: 4096 tokens
- **Training Framework**: Unsloth & PEFT (LoRA)
- **Quantization/Formats**: 
  - Full merged 16-bit `bfloat16`
  - GGUF (`q4_k_m`) for local inference (Ollama, LM Studio)

---

## 📚 Dataset and Domain Knowledge
The model has been continuously pre-trained and fine-tuned on a comprehensive corpus of ~40MB of high-quality Vedic and Ayurvedic literature:
1. **The Four Vedas (English Translations)**:
   - **Rig Veda** (Ralph T.H. Griffith translation)
   - **Sama Veda** (Ralph T.H. Griffith translation)
   - **Yajur Veda** (Arthur Berriedale Keith's Taittiriya/Black Yajur Veda & Griffith's Vajasaneya/White Yajur Veda translations)
   - **Atharva Veda** (Ralph T.H. Griffith translation)
2. **Ayurvedic Samhitas & Texts**:
   - **Charaka Samhita**: Ancient text on internal medicine, therapeutics, and diagnostics.
   - **Sushruta Samhita**: Ancient foundational text on Ayurvedic surgery and instruments.
   - **Rasa Jala Nidhi**: Comprehensive Ayurvedic treatise on Rasashastra (mineralology, alchemy, and chemistry).
   - **IRJAY (International Research Journal of Ayurveda and Yoga)**: Academic papers spanning clinical studies and theoretical frameworks of Ayurveda and Yoga.

---

## 🛠️ Usage Instructions

### 1. vLLM Serverless Serving (Recommended)
You can deploy and serve this model using `vllm` for blazing-fast inference with continuous batching and PagedAttention.
Here is a sample serving configuration:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="shinigamiRaj/IndicVedas",
    max_model_len=4096,
    dtype="bfloat16",
    trust_remote_code=True,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
)

sampling_params = SamplingParams(
    temperature=0.2,
    top_p=0.9,
    max_tokens=512,
    repetition_penalty=1.15,
    stop=["<|im_end|>", "<|endoftext|>"],
)

# Q&A Chat Prompt Structure (ChatML)
messages = [
    {
        "role": "system",
        "content": (
            "You are VedaGPT, an expert scholar of the ancient Vedic scriptures like RigVeda, SamaVeda, YajurVeda, AtharvaVeda, Charaka Samhita, Sushruta Samhita, Ayurveda, and Yoga. "
            "Answer questions accurately based on your knowledge of the Vedas, Upanishads, Charaka Samhita, Sushruta Samhita, and other classical Indian texts. "
            "Maintain the style of writing as per the ancient Vedic texts where required."
        )
    },
    {"role": "user", "content": "What are the key pillars of health according to Ayurveda?"}
]

tokenizer = llm.get_tokenizer()
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

outputs = llm.generate([formatted_prompt], sampling_params)
print(outputs[0].outputs[0].text)
```

### 2. Standard Transformers Loading
To load and run inference with Hugging Face `transformers` in 16-bit:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "shinigamiRaj/IndicVedas"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Example chat template inference:
messages = [
    {"role": "system", "content": "You are VedaGPT, an expert scholar of ancient texts."},
    {"role": "user", "content": "Tell me about Agni in the Rig Veda."}
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(model.device)
outputs = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True))
```

### 3. Local Deployment via Ollama (GGUF)
Since this repository includes the GGUF `q4_k_m` files, you can create an Ollama model locally. Create a `Modelfile`:

```text
FROM ./IndicVedas-Q4_K_M.gguf

TEMPLATE \"\"\"<|im_start|>system
You are VedaGPT, an expert scholar of the ancient Vedic scriptures like RigVeda, SamaVeda, YajurVeda, AtharvaVeda, Charaka Samhita, Sushruta Samhita, Ayurveda, and Yoga. Answer questions accurately based on your knowledge of the Vedas, Upanishads, Charaka Samhita, Sushruta Samhita, and other classical Indian texts. Maintain the style of writing as per the ancient Vedic texts where required.<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
\"\"\"

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.15
```

Build and run:
```bash
ollama create VedaGPT -f Modelfile
ollama run VedaGPT
```

---

## 🔬 Training Configuration
- **Hardware**: Modal Serverless Cloud GPU (NVIDIA L40S)
- **Quantization (during training)**: 4-bit NF4
- **Parameters**: 
  - PEFT Rank (`r`): 64
  - LoRA Alpha: 64
  - Optimizer: `adamw_8bit`
  - Learning Rate: `2e-5` with `cosine` scheduler
  - Epochs: 1
  - Max Sequence Length: 4096 tokens
"""

# 2. README.md for shinigamiRaj/IndicVedas-LoRA (PEFT Adapter)
lora_readme = """---
license: other
license_name: public-domain
language:
- en
- sa
- hi
tags:
- Vedas
- Ayurveda
- Qwen
- Qwen2.5
- unsloth
- lora
- peft
- Text-Generation
- Sanskrit
base_model: Qwen/Qwen2.5-14B-Instruct
pipeline_tag: text-generation
---

# 🪶 VedaGPT: LoRA Adapters (IndicVedas-LoRA)

This repository (`shinigamiRaj/IndicVedas-LoRA`) hosts the **PEFT LoRA adapter weights** for VedaGPT. 

VedaGPT is fine-tuned on a rich corpus of ancient Indian Vedic literature (Rig Veda, Sama Veda, Yajur Veda, Atharva Veda) and classic Ayurvedic medicine treatises (Charaka Samhita, Sushruta Samhita, Rasa Jala Nidhi, IRJAY papers).

The base model is **[Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)**, and these adapters were trained using Unsloth on serverless Modal GPUs.

---

## 🏛️ Adapter Details
- **Base Model**: `Qwen/Qwen2.5-14B-Instruct`
- **Adapter Type**: LoRA (Low-Rank Adaptation)
- **Max Sequence Length**: 4096 tokens
- **Training Framework**: Unsloth & PEFT
- **Parameters**:
  - Rank (`r`): 64
  - Alpha: 64
  - Target Modules: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
  - Rank-Stabilized LoRA (rsLoRA): True

---

## 🛠️ Usage Instructions

### 1. Loading with Unsloth (Fastest & Easiest)
Unsloth is highly recommended for running inference or further training on these adapters.

```python
from unsloth import FastLanguageModel

max_seq_length = 4096
dtype = None # None for auto-detection

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "shinigamiRaj/IndicVedas-LoRA", # Loads base model + adapter automatically
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = True, # Set to False for 16-bit
)
FastLanguageModel.for_inference(model)

# Generate using apply_chat_template
messages = [
    {
        "role": "system",
        "content": (
            "You are VedaGPT, an expert scholar of the ancient Vedic scriptures like RigVeda, SamaVeda, YajurVeda, AtharvaVeda, Charaka Samhita, Sushruta Samhita, Ayurveda, and Yoga. "
            "Answer questions accurately based on your knowledge of the Vedas, Upanishads, Charaka Samhita, Sushruta Samhita, and other classical Indian texts. "
            "Maintain the style of writing as per the ancient Vedic texts where required."
        )
    },
    {"role": "user", "content": "Tell me about the connection between Agni and Rig Veda Hymn 1."}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    repetition_penalty=1.15
)

response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
print(response)
```

### 2. Loading with standard PEFT & Transformers
If you aren't using Unsloth, load the base model and apply the adapters using the Hugging Face `peft` library.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-14B-Instruct"
adapter_id = "shinigamiRaj/IndicVedas-LoRA"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_id)

# Build messages and apply chat template...
```

---

## 🔬 Training Configuration
- **Hardware**: Modal Serverless Cloud GPU (NVIDIA L40S)
- **Quantization (during training)**: 4-bit NF4
- **Parameters**: 
  - PEFT Rank (`r`): 64
  - LoRA Alpha: 64
  - Optimizer: `adamw_8bit`
  - Learning Rate: `2e-5` with `cosine` scheduler
  - Epochs: 1
  - Max Sequence Length: 4096 tokens
"""

# Push to Hugging Face
print("Uploading README.md for shinigamiRaj/IndicVedas...")
api.upload_file(
    path_or_fileobj=merged_readme.encode("utf-8"),
    path_in_repo="README.md",
    repo_id="shinigamiRaj/IndicVedas",
    repo_type="model"
)
print("README.md uploaded successfully for shinigamiRaj/IndicVedas!")

print("Uploading README.md for shinigamiRaj/IndicVedas-LoRA...")
api.upload_file(
    path_or_fileobj=lora_readme.encode("utf-8"),
    path_in_repo="README.md",
    repo_id="shinigamiRaj/IndicVedas-LoRA",
    repo_type="model"
)
print("README.md uploaded successfully for shinigamiRaj/IndicVedas-LoRA!")
print("Done!")

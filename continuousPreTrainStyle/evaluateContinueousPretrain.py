import os
import json
import gc
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from colorama import Fore, init
from tqdm import tqdm

# Initialize colorama
init(autoreset=True)

# Script base directory for absolute relative path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration
BASE_MODEL_PATH = "/Users/raj/PycharmProjects/VedaGPT/base_model_local"
ADAPTER_PATH = os.path.join(SCRIPT_DIR, "outputs", "final_model")
TEST_FILE = os.path.join(SCRIPT_DIR, "data", "test_split.jsonl")
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

def load_test_data(file_path):
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            if "text" in data:
                texts.append(data["text"])
    return texts

def evaluate_perplexity(model, tokenizer, texts):
    """Calculate cross-entropy loss and perplexity on the test dataset."""
    total_loss = 0.0
    total_tokens = 0
    
    print(f"📊 Evaluating {len(texts)} test examples...")
    with torch.no_grad():
        for text in tqdm(texts, desc="Processing tokens", unit="doc"):
            if not text.strip():
                continue
                
            inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
            inputs["labels"] = inputs["input_ids"].clone()
            
            outputs = model(**inputs)
            loss = outputs.loss.item()
            num_tokens = inputs["input_ids"].numel()
            
            # Weighted loss by number of tokens in the document
            total_loss += loss * num_tokens
            total_tokens += num_tokens
            
    if total_tokens == 0:
        return float('inf'), float('inf')
        
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

def main():
    print(f"📊 {Fore.LIGHTCYAN_EX}Starting Continuous Pre-training Evaluation Suite on {DEVICE.upper()}...{Fore.RESET}")
    
    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"Test split not found at {TEST_FILE}. Make sure splits exist first.")
    
    texts = load_test_data(TEST_FILE)
    # Subset to 3 examples for fast evaluation validation
    texts = texts[:2]
    print(f"📝 Loaded {len(texts)} test records from {TEST_FILE} (Subsampled to 3 for speed).")
    
    # ----------------------------------------------------
    # Phase 1: Load and Run Base Model
    # ----------------------------------------------------
    print(f"\n📂 Loading base tokenizer and quantized base model from: {BASE_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    if DEVICE == "mps":
        print("💡 Apple Silicon detected: Loading model in high-speed float16 precision (Bypassing 4-bit quantization bottleneck)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            device_map=DEVICE,
            torch_dtype=torch.float16,
        )
    else:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if DEVICE != "cpu" else torch.float32,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            device_map=DEVICE,
            quantization_config=quant_config,
        )
    
    print("🚀 Evaluating Base Model Perplexity...")
    base_loss, base_ppl = evaluate_perplexity(base_model, tokenizer, texts)
    print(f"   Base Loss: {base_loss:.4f} | Perplexity: {base_ppl:.4f}")
    
    # Clean memory immediately
    print("🧹 Cleaning base model from memory...")
    del base_model
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ----------------------------------------------------
    # Phase 2: Load and Run Pre-trained Model
    # ----------------------------------------------------
    if not os.path.exists(ADAPTER_PATH):
        raise FileNotFoundError(f"LoRA adapters not found at '{ADAPTER_PATH}'. Run continuousPreTrainStyle/finetuneContinueousPretrain.py first!")
        
    print(f"\n📂 Loading base model with pre-trained LoRA adapters from: {ADAPTER_PATH}...")
    if DEVICE == "mps":
        base_model_for_peft = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            device_map=DEVICE,
            torch_dtype=torch.float16,
        )
    else:
        base_model_for_peft = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            device_map=DEVICE,
            quantization_config=quant_config,
        )
    
    peft_model = PeftModel.from_pretrained(base_model_for_peft, ADAPTER_PATH)
    peft_model.eval()
    
    print("🚀 Evaluating Pre-trained Model Perplexity...")
    pt_loss, pt_ppl = evaluate_perplexity(peft_model, tokenizer, texts)
    print(f"   Pre-trained Loss: {pt_loss:.4f} | Perplexity: {pt_ppl:.4f}")
    
    # Clean up
    print("🧹 Cleaning PEFT model from memory...")
    del peft_model
    del base_model_for_peft
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ----------------------------------------------------
    # Phase 3: Metrics Computation and Report
    # ----------------------------------------------------
    print(f"\n📈 {Fore.LIGHTMAGENTA_EX}Calculating Metrics Comparison...{Fore.RESET}")
    
    loss_diff = pt_loss - base_loss
    ppl_diff = pt_ppl - base_ppl
    ppl_improvement = ((base_ppl - pt_ppl) / base_ppl) * 100 if base_ppl != 0 else 0
    
    # Print comparative report
    print("\n" + "=" * 70)
    print(f"📊 {Fore.LIGHTGREEN_EX}CONTINUOUS PRE-TRAINING EVALUATION REPORT (BASE vs. PRE-TRAINED){Fore.RESET}")
    print("=" * 70)
    print(f"{'Metric':<20} | {'Base Model':<15} | {'Pre-trained Model':<15} | {'Improvement':<15}")
    print("-" * 70)
    
    loss_color = Fore.LIGHTGREEN_EX if loss_diff < 0 else Fore.LIGHTRED_EX
    ppl_color = Fore.LIGHTGREEN_EX if ppl_diff < 0 else Fore.LIGHTRED_EX
    
    print(f"{'Cross-Entropy Loss':<20} | {base_loss:<15.4f} | {pt_loss:<15.4f} | {loss_color}{loss_diff:+.4f}{Fore.RESET}")
    print(f"{'Perplexity (PPL)':<20} | {base_ppl:<15.4f} | {pt_ppl:<15.4f} | {ppl_color}{ppl_improvement:+.2f}%{Fore.RESET}")
    print("=" * 70)
    
    print(f"\n💡 {Fore.LIGHTCYAN_EX}Note: Lower Perplexity represents better native text prediction capability.{Fore.RESET}\n")

if __name__ == "__main__":
    main()

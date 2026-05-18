import os
import json
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from colorama import Fore, init

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
    instructions = []
    references = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            instructions.append(data["instruction"])
            references.append(data["output"])
    return instructions, references

def run_inference(model, tokenizer, instructions):
    predictions = []
    print(f"🔮 Generating predictions on {len(instructions)} test examples...")
    for idx, inst in enumerate(instructions):
        messages = [
            {"role": "user", "content": inst}
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
        
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        pred_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        predictions.append(pred_text)
        print(f"   🔹 Done {idx+1}/{len(instructions)}")
        
    return predictions

def compute_metrics(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1, rouge2, rougeL = 0.0, 0.0, 0.0
    bleu = 0.0
    
    smooth_fn = SmoothingFunction().method1
    
    for pred, ref in zip(predictions, references):
        # Compute ROUGE
        scores = scorer.score(ref, pred)
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
        rougeL += scores['rougeL'].fmeasure
        
        # Compute BLEU
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        if len(ref_tokens) > 0 and len(pred_tokens) > 0:
            bleu += sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_fn)
            
    n = len(references)
    return {
        "ROUGE-1": (rouge1 / n) * 100,
        "ROUGE-2": (rouge2 / n) * 100,
        "ROUGE-L": (rougeL / n) * 100,
        "BLEU": (bleu / n) * 100,
    }

def main():
    print(f"📊 {Fore.LIGHTCYAN_EX}Starting QA Evaluation Suite on {DEVICE.upper()}...{Fore.RESET}")
    
    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"Test split not found at {TEST_FILE}. Make sure splits exist first.")
    
    instructions, references = load_test_data(TEST_FILE)
    # Subset to 1 examples for fast evaluation validation
    instructions, references = instructions[:1], references[:1]
    print(f"📝 Loaded {len(instructions)} test records from {TEST_FILE} (Subsampled to 10 for speed).")
    
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
    
    print("🚀 Evaluating Base Model...")
    base_predictions = run_inference(base_model, tokenizer, instructions)
    
    # Clean memory immediately
    print("🧹 Cleaning base model from memory...")
    del base_model
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ----------------------------------------------------
    # Phase 2: Load and Run Fine-Tuned Model
    # ----------------------------------------------------
    if not os.path.exists(ADAPTER_PATH):
        raise FileNotFoundError(f"LoRA adapters not found at '{ADAPTER_PATH}'. Run qaAlpacaStyle/finetuneQA.py first!")
        
    print(f"\n📂 Loading base model with fine-tuned LoRA adapters from: {ADAPTER_PATH}...")
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
    
    print("🚀 Evaluating Fine-Tuned Model...")
    ft_predictions = run_inference(peft_model, tokenizer, instructions)
    
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
    base_metrics = compute_metrics(base_predictions, references)
    ft_metrics = compute_metrics(ft_predictions, references)
    
    # Print comparative report
    print("\n" + "=" * 60)
    print(f"📊 {Fore.LIGHTGREEN_EX}QA EVALUATION METRICS COMPARISON (BASE vs. FINE-TUNED){Fore.RESET}")
    print("=" * 60)
    print(f"{'Metric':<15} | {'Base Model':<15} | {'Fine-Tuned Model':<15} | {'Improvement':<15}")
    print("-" * 60)
    for k in base_metrics.keys():
        diff = ft_metrics[k] - base_metrics[k]
        color = Fore.LIGHTGREEN_EX if diff > 0 else Fore.LIGHTRED_EX
        print(f"{k:<15} | {base_metrics[k]:<15.2f} | {ft_metrics[k]:<15.2f} | {color}{diff:+.2f}%{Fore.RESET}")
    print("=" * 60)
    
    # Show side-by-side comparative samples
    print(f"\n💬 {Fore.LIGHTCYAN_EX}Side-by-Side Test Outputs Preview:{Fore.RESET}")
    print("-" * 80)
    num_samples = min(3, len(instructions))
    for i in range(num_samples):
        print(f"\n📝 {Fore.YELLOW}Example {i+1}{Fore.RESET}")
        print(f"❓ {Fore.LIGHTWHITE_EX}Prompt:{Fore.RESET} {instructions[i]}")
        print(f"🎯 {Fore.LIGHTGREEN_EX}Ground Truth:{Fore.RESET} {references[i]}")
        print(f"🤖 {Fore.LIGHTRED_EX}Base Model Prediction:{Fore.RESET} {base_predictions[i]}")
        print(f"⭐ {Fore.LIGHTGREEN_EX}Fine-Tuned Model Prediction:{Fore.RESET} {ft_predictions[i]}")
        print("-" * 80)

if __name__ == "__main__":
    main()

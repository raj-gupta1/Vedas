import os
import json
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from colorama import Fore, init

init(autoreset=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_MODEL_PATH = os.path.join(SCRIPT_DIR, "base_model_local")
QA_ADAPTER_PATH = os.path.join(SCRIPT_DIR, "qaAlpacaStyle", "outputs", "final_model")
CPT_ADAPTER_PATH = os.path.join(SCRIPT_DIR, "continuousPreTrainStyle", "outputs", "final_model")
TEST_FILE = os.path.join(SCRIPT_DIR, "qaAlpacaStyle", "data", "test_split.jsonl")
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

def load_test_data(file_path):
    instructions = []
    references = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            # For QA dataset
            if "instruction" in data and "output" in data:
                instructions.append(data["instruction"])
                references.append(data["output"])
            # Fallback for continuous pretrain data format
            elif "text" in data:
                # We can't really do QA eval on raw text without prompt/response separation, 
                # but we will extract something if needed.
                pass
    return instructions, references

def run_inference(model, tokenizer, instructions):
    predictions = []
    print(f"🔮 Generating predictions on {len(instructions)} test examples...")
    for idx, inst in enumerate(instructions):
        messages = [{"role": "user", "content": inst}]
        
        # Check if tokenizer has chat template, if not manually format
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(DEVICE)
        else:
            prompt = f"Instruction: {inst}\nOutput:"
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
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
    rouge1, rouge2, rougeL, bleu = 0.0, 0.0, 0.0, 0.0
    smooth_fn = SmoothingFunction().method1
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
        rougeL += scores['rougeL'].fmeasure
        
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        if len(ref_tokens) > 0 and len(pred_tokens) > 0:
            bleu += sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_fn)
            
    n = max(len(references), 1)
    return {
        "ROUGE-1": (rouge1 / n) * 100,
        "ROUGE-2": (rouge2 / n) * 100,
        "ROUGE-L": (rougeL / n) * 100,
        "BLEU": (bleu / n) * 100,
    }

def get_base_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    if DEVICE == "mps":
        model = AutoModelForCausalLM.from_pretrained(
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
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            device_map=DEVICE,
            quantization_config=quant_config,
        )
    return model, tokenizer

def clean_memory(model):
    del model
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()

def main():
    print(f"📊 {Fore.LIGHTCYAN_EX}Starting Unified Evaluation Suite on {DEVICE.upper()}...{Fore.RESET}")
    
    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"Test split not found at {TEST_FILE}.")
    
    instructions, references = load_test_data(TEST_FILE)
    instructions, references = instructions[:3], references[:3]  # Subsample for speed
    print(f"📝 Loaded {len(instructions)} QA test records from {TEST_FILE}.")
    
    results = {}
    
    # 1. Base Model
    print(f"\n📂 Loading BASE MODEL...")
    base_model, tokenizer = get_base_model_and_tokenizer()
    preds_base = run_inference(base_model, tokenizer, instructions)
    results["Base"] = {"metrics": compute_metrics(preds_base, references), "preds": preds_base}
    clean_memory(base_model)
    
    # 2. Continuous Pre-Trained Model
    if os.path.exists(CPT_ADAPTER_PATH):
        print(f"\n📂 Loading CONTINUOUS PRE-TRAINED MODEL (CPT)...")
        base_model_for_peft, _ = get_base_model_and_tokenizer()
        cpt_model = PeftModel.from_pretrained(base_model_for_peft, CPT_ADAPTER_PATH)
        cpt_model.eval()
        preds_cpt = run_inference(cpt_model, tokenizer, instructions)
        results["CPT"] = {"metrics": compute_metrics(preds_cpt, references), "preds": preds_cpt}
        clean_memory(cpt_model)
        clean_memory(base_model_for_peft)
    else:
        print(f"⚠️ CPT adapter not found at {CPT_ADAPTER_PATH}. Skipping.")
        
    # 3. QA Alpaca-Style Model
    if os.path.exists(QA_ADAPTER_PATH):
        print(f"\n📂 Loading QA ALPACA-STYLE MODEL...")
        base_model_for_peft, _ = get_base_model_and_tokenizer()
        qa_model = PeftModel.from_pretrained(base_model_for_peft, QA_ADAPTER_PATH)
        qa_model.eval()
        preds_qa = run_inference(qa_model, tokenizer, instructions)
        results["QA"] = {"metrics": compute_metrics(preds_qa, references), "preds": preds_qa}
        clean_memory(qa_model)
        clean_memory(base_model_for_peft)
    else:
        print(f"⚠️ QA adapter not found at {QA_ADAPTER_PATH}. Skipping.")

    # Print comparative report
    print("\n" + "=" * 80)
    print(f"📊 {Fore.LIGHTGREEN_EX}UNIFIED EVALUATION METRICS COMPARISON{Fore.RESET}")
    print("=" * 80)
    models = list(results.keys())
    header = f"{'Metric':<15} | " + " | ".join([f"{m:<15}" for m in models])
    print(header)
    print("-" * 80)
    
    metrics_keys = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"]
    for k in metrics_keys:
        row = f"{k:<15} | " + " | ".join([f"{results[m]['metrics'][k]:<15.2f}" for m in models])
        print(row)
    print("=" * 80)
    
    print(f"\n💬 {Fore.LIGHTCYAN_EX}Side-by-Side Test Outputs Preview:{Fore.RESET}")
    print("-" * 80)
    for i in range(len(instructions)):
        print(f"\n📝 {Fore.YELLOW}Example {i+1}{Fore.RESET}")
        print(f"❓ {Fore.LIGHTWHITE_EX}Prompt:{Fore.RESET} {instructions[i]}")
        print(f"🎯 {Fore.LIGHTGREEN_EX}Ground Truth:{Fore.RESET} {references[i]}")
        for m in models:
            color = Fore.LIGHTRED_EX if m == "Base" else (Fore.LIGHTBLUE_EX if m == "CPT" else Fore.LIGHTGREEN_EX)
            print(f"🤖 {color}{m} Prediction:{Fore.RESET} {results[m]['preds'][i]}")
        print("-" * 80)

if __name__ == "__main__":
    main()

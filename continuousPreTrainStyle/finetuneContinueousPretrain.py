import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from colorama import Fore, init

# Initialize colorama
init(autoreset=True)

# Script base directory for absolute relative path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# Configuration and Constants
# ==========================================
base_model_local = "/Users/raj/PycharmProjects/VedaGPT/base_model_local"
base_model_hf = "dheeyantra/dhee-nxtgen-qwen3-indic"
train_file = os.path.join(SCRIPT_DIR, "data", "train_split.jsonl")
val_file = os.path.join(SCRIPT_DIR, "data", "val_split.jsonl")
hf_token = os.getenv("HF_TOKEN", "hf access token here")

# Determine device map based on hardware (Apple Silicon M4 uses mps:0, CUDA uses cuda:0)
has_cuda = torch.cuda.is_available()
has_mps = torch.backends.mps.is_available()
device_map = "mps:0" if has_mps else ("cuda:0" if has_cuda else "cpu")

# ==========================================
# Custom Progress and Checkpoint Callback
# ==========================================
class CustomProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.max_steps is not None and state.max_steps > 0:
            current_step = state.global_step
            total_steps = state.max_steps
            percent = (current_step / total_steps) * 100
            epoch = state.epoch if state.epoch is not None else 0.0
            print(f"\n📊 {Fore.LIGHTCYAN_EX}Pre-Training Progress: {current_step}/{total_steps} steps ({percent:.2f}%) | Epoch: {epoch:.2f}{Fore.RESET}")
            if logs:
                if "loss" in logs:
                    print(f"   📉 {Fore.LIGHTYELLOW_EX}Training Loss: {logs['loss']:.4f}{Fore.RESET}")
                if "eval_loss" in logs:
                    print(f"   🎯 {Fore.LIGHTGREEN_EX}Validation Loss: {logs['eval_loss']:.4f}{Fore.RESET}")

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        print(f"\n💾 {Fore.LIGHTGREEN_EX}Checkpoint successfully saved at step {state.global_step} to: {checkpoint_dir}{Fore.RESET}\n")

def main():
    print(f"🚀 {Fore.LIGHTCYAN_EX}Starting Continuous Pre-Training Setup...{Fore.RESET}")
    print(f"💻 Device Map Configured to: {device_map}")

    # Determine load source based on whether base_model_local exists
    local_exists = os.path.exists(base_model_local) and len(os.listdir(base_model_local)) > 0
    if local_exists:
        print(f"✅ {Fore.LIGHTGREEN_EX}Found shared base model cache at: '{base_model_local}'. Loading offline!{Fore.RESET}")
        model_load_source = base_model_local
    else:
        print(f"📂 {Fore.LIGHTYELLOW_EX}Shared base model cache not found. Downloading from Hugging Face: '{base_model_hf}'...{Fore.RESET}")
        model_load_source = base_model_hf

    # ==========================================
    # Tokenizer Loading
    # ==========================================
    print(f"📥 Loading tokenizer from: {model_load_source}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_load_source, 
        trust_remote_code=True,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ==========================================
    # Dataset Loading
    # ==========================================
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        raise FileNotFoundError("Pre-train split files not found! Make sure to run 'splitContinueousPretrainData.py' inside the continuousPreTrainStyle directory first.")

    print(f"📂 Loading split datasets: {train_file} and {val_file}...")
    train_dataset = load_dataset("json", data_files=train_file)["train"]
    val_dataset = load_dataset("json", data_files=val_file)["train"]
    
    print("\n🔍 Raw Pre-Training Text Sample:")
    print(Fore.LIGHTMAGENTA_EX + str(train_dataset[0]) + Fore.RESET + "\n")

    # ==========================================
    # Quantization Configuration
    # ==========================================
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if (has_cuda or has_mps) else torch.float32,
    )

    # ==========================================
    # Model Loading
    # ==========================================
    print(f"📥 Loading quantized base model from: {model_load_source}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_load_source,
        device_map=device_map,
        quantization_config=quant_config,
        token=hf_token,
        cache_dir="./workspace" if not local_exists else None
    )

    # Save to local if downloaded online for the first time
    if not local_exists:
        print(f"💾 Saving downloaded base model locally to: '{base_model_local}' for future runs...")
        os.makedirs(base_model_local, exist_ok=True)
        model.save_pretrained(base_model_local)
        tokenizer.save_pretrained(base_model_local)
        print(f"✅ {Fore.LIGHTGREEN_EX}Base model and tokenizer successfully saved locally at '{base_model_local}'!{Fore.RESET}")

    # Gradient Checkpointing and PEFT Prep (User Style)
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # ==========================================
    # LoRA / PEFT Configuration
    # ==========================================
    peft_config = LoraConfig(
        r=256,
        lora_alpha=512,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    # ==========================================
    # Trainer Setup and Execution (Isolated under outputs/)
    # ==========================================
    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            output_dir=os.path.join(SCRIPT_DIR, "outputs", "checkpoints"), 
            num_train_epochs=3,  
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            logging_steps=5,
            learning_rate=2e-5,  
            save_strategy="steps",
            save_steps=50,       
            save_total_limit=3,   
            eval_strategy="steps",
            eval_steps=50,       
            per_device_eval_batch_size=1,
            report_to="none",
            dataset_text_field="text",
            max_length=2048,
        ),
        peft_config=peft_config,
        callbacks=[CustomProgressCallback()]
    )

    # Train
    print("🔥 Starting Continuous Pre-Training loop...")
    trainer.train()

    # Save complete models
    trainer.save_model(os.path.join(SCRIPT_DIR, 'outputs', 'complete_checkpoint'))
    trainer.model.save_pretrained(os.path.join(SCRIPT_DIR, "outputs", "final_model"))
    print(f"🎉 Done! Models saved to absolute paths inside continuousPreTrainStyle/outputs/.")

if __name__ == "__main__":
    main()

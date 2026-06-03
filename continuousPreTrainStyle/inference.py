import os
import gc
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from colorama import Fore, init

# Load environment variables
load_dotenv()

# Authenticate with Hugging Face Hub to prevent rate limits and speed up downloads
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

init(autoreset=True)

# Configuration
BASE_MODEL_PATH = "unsloth/Llama-3.2-3B-Instruct" 
ADAPTER_PATH = "/Users/raj/PycharmProjects/VedaGPT/continuousPreTrainStyle/lora_model"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"🚀 {Fore.LIGHTCYAN_EX}Starting Inference on {DEVICE.upper()}...{Fore.RESET}")

    # Load Tokenizer
    print(f"📂 Loading tokenizer from: {BASE_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Base Model
    print(f"📂 Loading base model from: {BASE_MODEL_PATH}...")
    if DEVICE == "mps":
        print("💡 Apple Silicon detected: Loading model in high-speed float16 precision...")
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
            quantization_config=quant_config if DEVICE != "cpu" else None,
        )

    # Load LoRA Adapter
    if not os.path.exists(ADAPTER_PATH):
        raise FileNotFoundError(f"LoRA adapters not found at '{ADAPTER_PATH}'.")

    print(f"📂 Loading LoRA adapters from: {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    
    print(f"\n{Fore.LIGHTGREEN_EX}✅ Model loaded successfully! Type 'quit' or 'exit' to stop.{Fore.RESET}\n")

    while True:
        try:
            prompt = input(f"{Fore.LIGHTYELLOW_EX}Enter your prompt: {Fore.RESET}")
            if prompt.lower() in ["quit", "exit"]:
                break
            if not prompt.strip():
                continue

            # Format the prompt using the model's chat template
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
                messages = [
                    {"role": "system", "content": "You are VedaGPT, a wise and knowledgeable AI assistant deeply familiar with the ancient Indian Vedas (Rig, Sama, Yajur, and Atharva). Answer questions faithfully, accurately, and respectfully based on Vedic texts."},
                    {"role": "user", "content": prompt}
                ]
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                formatted_prompt = prompt + "\n"

            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    temperature=0.5,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.15, # Helps prevent repeating the exact same sentence
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode the generated text (skip the prompt)
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            print(f"\n{Fore.LIGHTBLUE_EX}Model Response:{Fore.RESET}")
            print(response.strip())
            print("-" * 70)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"{Fore.RED}Error during generation: {e}{Fore.RESET}")

if __name__ == "__main__":
    main()

"""
Efficient LLM Finetuning with Unsloth on Modal
================================================
Converted from qlorafinetuneV2.ipynb for use with `modal run --detach`.

Usage:
    source continuousPreTrainStyle/vedaFineTune/bin/activate
    modal run --detach continuousPreTrainStyle/qlorafinetunev2.py

To run only training:
    modal run --detach continuousPreTrainStyle/qlorafinetunev2.py::train

To run only inference:
    modal run --detach continuousPreTrainStyle/qlorafinetunev2.py::infer

To run push & GGUF export:
    modal run --detach continuousPreTrainStyle/qlorafinetunev2.py::push_export
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import modal

app = modal.App("vedagpt-unsloth-finetune")

# Persistent volumes for caching model weights and saving checkpoints
model_cache_volume = modal.Volume.from_name("unsloth-model-cache", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("unsloth-checkpoints", create_if_missing=True)

# ---------------------------------------------------------------------------
# Container Image
# ---------------------------------------------------------------------------
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "cmake", "build-essential")
    .uv_pip_install(
        "accelerate==1.9.0",
        "datasets==3.6.0",
        "hf-transfer==0.1.9",
        "huggingface_hub==0.34.2",
        "peft==0.16.0",
        "transformers==4.54.0",
        "trl==0.19.1",
        "unsloth[cu128-torch270]==2025.7.8",
        "unsloth_zoo==2025.7.10",
        "wandb==0.21.0",
    )
    .env({"HF_HOME": "/model_cache", "HF_HUB_ENABLE_HF_TRANSFER": "1", "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"})
    .add_local_file(
        "continuousPreTrainStyle/data/continueousPreTrainData.jsonl",
        "/root/data/continueousPreTrainData.jsonl"
    )
)



# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def chunk_dataset_tokenized(dataset_list, tokenizer, target_chunk_size=4096, overlap=200):
    """Split long rows into overlapping token-level chunks.

    Preserves the Vedic header block [[ ... ]] across continuation chunks.
    """
    chunked_dataset = []
    long_count = 0

    for row in dataset_list:
        text = row["text"]
        tokens = tokenizer.encode(text)

        if len(tokens) <= target_chunk_size:
            chunked_dataset.append(row)
        else:
            long_count += 1
            header = ""
            if text.startswith("[[ "):
                header_end = text.find(" ]]")
                if header_end != -1:
                    header = text[:header_end + 3]

            stride = target_chunk_size - overlap
            for chunk_idx, i in enumerate(range(0, len(tokens), stride)):
                chunk_tokens = tokens[i : i + target_chunk_size]
                if len(chunk_tokens) < 50:
                    break
                chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

                if chunk_idx > 0 and header and not chunk_text.startswith("[[ "):
                    chunk_text = header + "\n\n" + chunk_text

                chunked_dataset.append({"text": chunk_text})

    print(f"  📊 Rows that needed chunking: {long_count}")
    print(f"  📊 Original rows: {len(dataset_list)} → Chunked rows: {len(chunked_dataset)}")
    return chunked_dataset


# ---------------------------------------------------------------------------
# Training Function
# ---------------------------------------------------------------------------
@app.function(
    image=train_image,
    gpu="L40S",
    cpu=2.0,
    memory=32768,
    timeout=6 * 3600,
    volumes={
        "/model_cache": model_cache_volume,
        "/checkpoints": checkpoint_volume,
    }
)
def train_remote():
    import json
    import random
    import torch
    import datasets
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    # 1. Load the Model
    max_seq_length = 4096
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )

    # 2. Configure PEFT/LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = 64,
        lora_alpha = 64,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = True,
        loftq_config = None,
    )

    # 3. Setup Tokenizer for CLM
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Load & Chunk Dataset
    print("Loading dataset from mounted local data...")
    dataset_list = []
    with open("/root/data/continueousPreTrainData.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset_list.append(json.loads(line))

    print(f"Loaded {len(dataset_list)} rows. Chunking dataset...")
    chunked_all = chunk_dataset_tokenized(dataset_list, tokenizer, target_chunk_size=max_seq_length, overlap=200)

    # Shuffle and split into train/val (90/10)
    random.seed(42)
    random.shuffle(chunked_all)
    val_size = max(1, int(len(chunked_all) * 0.1))
    chunked_val = chunked_all[:val_size]
    chunked_train = chunked_all[val_size:]

    print(f"After split: train={len(chunked_train)}, val={len(chunked_val)}")

    dataset_train = datasets.Dataset.from_list(chunked_train)
    dataset_train = dataset_train.map(lambda x: {"text": x["text"].strip() + tokenizer.eos_token}, batched=False)

    dataset_val = datasets.Dataset.from_list(chunked_val)
    dataset_val = dataset_val.map(lambda x: {"text": x["text"].strip() + tokenizer.eos_token}, batched=False)

    # 5. SFT Trainer Setup
    use_bf16 = torch.cuda.is_bf16_supported()

    import transformers
    from transformers import TrainerCallback
    transformers.utils.logging.set_verbosity_info()

    class PrinterCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                epoch = logs.get("epoch", state.epoch)
                loss = logs.get("loss", None)
                eval_loss = logs.get("eval_loss", None)
                if loss is not None:
                    print(f"Step {state.global_step}/{state.max_steps} | Epoch {epoch:.2f} | Loss: {loss:.4f}")
                if eval_loss is not None:
                    print(f"Step {state.global_step}/{state.max_steps} | Eval Loss: {eval_loss:.4f}")

    trainer = SFTTrainer(
        model = model,
        train_dataset = dataset_train,
        eval_dataset = dataset_val,
        processing_class = tokenizer,
        callbacks = [PrinterCallback()],
        args = SFTConfig(
            dataset_text_field = "text",
            max_seq_length = max_seq_length,
            dataset_num_proc = 2,
            packing = False,
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 2,
            warmup_steps = 50,
            num_train_epochs = 1,
            learning_rate = 2e-5,
            lr_scheduler_type = "cosine",
            weight_decay = 0.01,
            fp16 = not use_bf16,
            bf16 = use_bf16,
            logging_steps = 5,
            eval_strategy = "steps",
            eval_steps = 100,
            optim = "adamw_8bit",
            seed = 3407,
            report_to = "none",
            disable_tqdm = False,
            output_dir = "/checkpoints/outputs",
            save_strategy = "steps",
            save_steps = 100,
            save_total_limit = 3,
            load_best_model_at_end = True,
            metric_for_best_model = "eval_loss",
        ),
    )

    # 6. Run Training
    print("🔥 Starting pretraining...")
    model.for_training()
    trainer.train()

    # 7. Save model and tokenizer
    print("💾 Saving model adapters and tokenizer to Volume...")
    model.save_pretrained("/checkpoints/lora_model")
    tokenizer.save_pretrained("/checkpoints/lora_model")
    print("✅ Adapters and tokenizer saved to '/checkpoints/lora_model'!")


# ---------------------------------------------------------------------------
# Inference Function
# ---------------------------------------------------------------------------
@app.function(
    image=train_image,
    gpu="L40S",
    cpu=2.0,
    memory=32768,
    volumes={
        "/checkpoints": checkpoint_volume,
    }
)
def run_inference_remote():
    import os
    import glob
    import torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    results = []

    model_path = "/checkpoints/lora_model"
    if not os.path.exists(model_path):
        checkpoint_dirs = glob.glob("/checkpoints/outputs/checkpoint-*")
        if checkpoint_dirs:
            model_path = max(checkpoint_dirs, key=os.path.getctime)
        else:
            model_path = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"

    results.append(f"Loading model for inference from: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 4096,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    # MODE 1: Text Completion
    def complete_vedic_text(prompt, max_new_tokens=256):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            use_cache = True,
            temperature = 0.7,
            do_sample = True,
            top_p = 0.9,
            repetition_penalty = 1.15,
        )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        return response.strip()

    # MODE 2: Q&A Chat
    chat_tokenizer = get_chat_template(tokenizer, chat_template='chatml')

    def ask_vedagpt(query, max_new_tokens=256):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are VedaGPT, an expert scholar of the ancient Vedic scriptures like RigVeda, SamaVeda, YajurVeda, AtharvaVeda, Charaka Samhita, Sushruta Samhita, Ayurveda, and Yoga."
                           "Answer questions accurately based on your knowledge of the Vedas, Upanishads, Charaka Samhita, Sushruta Samhita, and other classical Indian texts."
                            "Maintain the style of writing as per the ancient Vedic texts where required."
                ),
            },
            {"role": "user", "content": query},
        ]
        inputs = chat_tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to("cuda")
        outputs = model.generate(
            input_ids = inputs,
            max_new_tokens = max_new_tokens,
            use_cache = True,
            temperature = 0.7,
            do_sample = True,
            top_p = 0.9,
            repetition_penalty = 1.15,
        )
        response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        return response.strip()

    # Test Text Completion
    results.append("\n" + "="*60)
    results.append("TEXT COMPLETION TESTS")
    results.append("="*60)
    completion_prompts = [
        "[[ Collection: Rig Veda | Translator: Ralph T.H. Griffith | Book: 1 | Hymn: HYMN I | Title: Agni ]]\n\n",
        "[[ Collection: Atharva Veda | Translator: Ralph T.H. Griffith | Book: 1 | Hymn: HYMN I ]]\n\n",
        "[[ Collection: Charaka Samhita ]]\n\n",
    ]
    for prompt in completion_prompts:
        results.append(f"\nPROMPT: {prompt[:80].strip()}...")
        results.append(f"COMPLETION: {complete_vedic_text(prompt)[:300]}")

    # Test Q&A Chat
    results.append("\n" + "="*60)
    results.append("Q&A CHAT TESTS")
    results.append("="*60)
    qa_queries = [
        "What are the four Vedas?",
        "Recite Hymn 1 from the Rig Veda.",
        "What does the Charaka Samhita say about digestion?",
    ]
    for q in qa_queries:
        results.append(f"\nQ: {q}")
        results.append(f"A: {ask_vedagpt(q)}")

    return "\n".join(results)


# ---------------------------------------------------------------------------
# Push & GGUF Export Function
# ---------------------------------------------------------------------------
@app.function(
    image=train_image,
    gpu="L40S",
    cpu=2.0,
    memory=32768,
    timeout=2 * 3600,
    volumes={
        "/checkpoints": checkpoint_volume,
    }
)
def push_and_export_remote():
    import os
    import glob
    import torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    hf_token = "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"

    model_path = "/checkpoints/lora_model"
    if not os.path.exists(model_path):
        checkpoint_dirs = glob.glob("/checkpoints/outputs/checkpoint-*")
        if checkpoint_dirs:
            model_path = max(checkpoint_dirs, key=os.path.getctime)
        else:
            raise FileNotFoundError("Could not find any checkpoints or saved model.")

    print(f"✅ Loading model from path: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 16384,
        dtype = torch.bfloat16,
        load_in_4bit = False,  # Changed to False to allow weight merging/GGUF export
    )

    tokenizer = get_chat_template(tokenizer, chat_template="chatml")
    print("✅ Qwen ChatML template re-applied.")

    # 1. Push LoRA adapters
    print("Pushing LoRA adapters to HF...")
    model.push_to_hub("shinigamiRaj/IndicVedas-LoRA", token=hf_token)
    tokenizer.push_to_hub("shinigamiRaj/IndicVedas-LoRA", token=hf_token)

    # 2. Push merged 16-bit model
    print("Pushing merged 16-bit model to HF...")
    model.push_to_hub_merged(
        "shinigamiRaj/IndicVedas",
        tokenizer,
        save_method = "merged_16bit",
        token = hf_token,
    )

    # 3. Export and Push GGUF straight into that same main base repo
    print("Exporting and pushing GGUF for local use...")
    model.push_to_hub_gguf(
        "shinigamiRaj/IndicVedas",
        tokenizer,
        quantization_method = "q4_k_m",
        token = hf_token,
    )

    # 4. Export GGUF locally to the volume copy as backup
    print("Exporting GGUF for local use...")
    model.save_pretrained_gguf(
        "/checkpoints/IndicVedas_GGUF",
        tokenizer,
        quantization_method = "q4_k_m",
    )
    print("🎉 All formats successfully exported and uploaded!")

# ---------------------------------------------------------------------------
# Local Entrypoints (for `modal run`)
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def train():
    """Run the training job."""
    print("🚀 Launching training on Modal...")
    train_remote.remote()
    print("✅ Training complete!")


@app.local_entrypoint()
def infer():
    """Run inference tests."""
    print("🚀 Launching inference on Modal...")
    output = run_inference_remote.remote()
    print(output)


@app.local_entrypoint()
def push_export():
    """Push to HF and export GGUF."""
    print("🚀 Launching push & export on Modal...")
    push_and_export_remote.remote()
    print("✅ Push & export complete!")

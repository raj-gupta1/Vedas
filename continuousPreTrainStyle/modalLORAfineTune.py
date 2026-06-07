"""
LoRA-Resumed Finetuning on Modal — Override Stale Knowledge (14B, Entire Dataset)
================================================================================
Resumes training from an existing 14B LoRA adapter (shinigamiRaj/IndicVedas-LoRA)
on the Qwen2.5-14B base model.

Strategy for maximum knowledge override:
  • 2 full epochs so every row is seen twice
  • Trained on 100% of the dataset (no train/val split)
  • Higher learning rate (5e-5) with linear warm-up then cosine decay
  • Weight decay (0.05) for generalization
  • Resumes the existing adapter directly (no get_peft_model call to avoid TypeError)
  • Optimized for L40S GPU (14B model)

Usage:
    modal run --detach continuousPreTrainStyle/modalLORAfineTune.py::train   # train
    modal run --detach continuousPreTrainStyle/modalLORAfineTune.py::infer   # inference
    modal run --detach continuousPreTrainStyle/modalLORAfineTune.py::push_export  # push & GGUF
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import modal

app = modal.App("vedagpt-lora-finetune-v2")

# Persistent volumes
model_cache_volume = modal.Volume.from_name("unsloth-model-cache", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("unsloth-checkpoints-v2", create_if_missing=True)

# ---------------------------------------------------------------------------
# Container Image — uses L40S for 14B model
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
    .env({
        "HF_HOME": "/model_cache",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
    })
    .add_local_file(
        "continuousPreTrainStyle/data/continueousPreTrainData.jsonl",
        "/root/data/continueousPreTrainData.jsonl",
    )
)


# ---------------------------------------------------------------------------
# Helper — token-level chunking with header preservation
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
# Training Function — Resume from existing LoRA on 14B base using L40S
# ---------------------------------------------------------------------------
@app.function(
    image=train_image,
    gpu="L40S",                 # 14B model runs beautifully on L40S
    cpu=2.0,
    memory=32768,               # 32 GB system RAM
    timeout=12 * 3600,          # 12 hours max
    volumes={
        "/model_cache": model_cache_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[modal.Secret.from_dotenv()],
)
def train_remote():
    import json
    import random
    import torch
    import datasets
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    # ── 1. Load the 14B base + existing LoRA adapter ──────────────────────
    # NOTE: Using max_seq_length = 4096 for training to fit in L40S VRAM (48GB).
    # 16384 is supported for inference, but during training it uses too much VRAM
    # and would result in Out of Memory (OOM) errors.
    max_seq_length = 4096

    print("📥 Loading 14B model with existing LoRA adapter shinigamiRaj/IndicVedas-LoRA...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="shinigamiRaj/IndicVedas-LoRA",  # loads base + adapter
        max_seq_length=max_seq_length,
        dtype=None,               # auto-detect
        load_in_4bit=True,
    )

    # ── 2. Configure for Training ─────────────────────────────────────────
    # We do NOT call FastLanguageModel.get_peft_model() because the model loaded above
    # is already a PeftModel. Calling get_peft_model() on an existing adapter raises a TypeError.
    # We call model.for_training() to prepare the loaded adapter for training.
    model.for_training()

    # ── 3. Tokenizer setup ────────────────────────────────────────────────
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 4. Load & chunk the FULL dataset ──────────────────────────────────
    print("📄 Loading complete dataset...")
    dataset_list = []
    with open("/root/data/continueousPreTrainData.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset_list.append(json.loads(line))

    print(f"  Loaded {len(dataset_list)} rows. Chunking...")
    chunked_all = chunk_dataset_tokenized(
        dataset_list, tokenizer,
        target_chunk_size=max_seq_length,
        overlap=200,
    )

    # Shuffle deterministically
    random.seed(42)
    random.shuffle(chunked_all)

    # Train on the entire dataset (no train/eval split)
    print(f"  Training dataset size: {len(chunked_all)}")

    dataset_train = datasets.Dataset.from_list(chunked_all)
    dataset_train = dataset_train.map(
        lambda x: {"text": x["text"].strip() + tokenizer.eos_token},
        batched=False,
    )

    # ── 5. SFT Trainer — aggressive 2-epoch config ───────────────────────
    use_bf16 = torch.cuda.is_bf16_supported()

    import transformers
    from transformers import TrainerCallback
    transformers.utils.logging.set_verbosity_info()

    class PrinterCallback(TrainerCallback):
        """Live logging of train loss to Modal stdout."""
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                epoch = logs.get("epoch", state.epoch)
                loss = logs.get("loss", None)
                lr = logs.get("learning_rate", None)
                if loss is not None:
                    lr_str = f" | LR: {lr:.2e}" if lr is not None else ""
                    print(
                        f"Step {state.global_step}/{state.max_steps} "
                        f"| Epoch {epoch:.2f} | Loss: {loss:.4f}{lr_str}"
                    )

    class VolumeCommitCallback(TrainerCallback):
        """Commit the checkpoint volume after each save so it's visible in the Modal dashboard."""
        def on_save(self, args, state, control, **kwargs):
            print(f"💾 Committing checkpoint volume at step {state.global_step}...")
            checkpoint_volume.commit()
            print(f"✅ Volume committed.")

    # ── Effective batch size: per_device (4) × grad_accum (2) = 8
    # ── With ~7k rows chunked, 2 epochs ≈ 1750 steps
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_train,
        processing_class=tokenizer,
        callbacks=[PrinterCallback(), VolumeCommitCallback()],
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=2,
            packing=False,

            # ── Batch sizing ──
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,

            # ── Schedule ──
            num_train_epochs=2,                 # 2 FULL epochs
            warmup_ratio=0.05,                  # 5% warm-up
            learning_rate=5e-5,                 # higher LR for stronger override
            lr_scheduler_type="cosine",

            # ── Regularisation ──
            weight_decay=0.05,                  # penalise large weights
            max_grad_norm=1.0,                  # gradient clipping for stability

            # ── Precision ──
            fp16=not use_bf16,
            bf16=use_bf16,

            # ── Logging & saving ──
            logging_steps=5,
            eval_strategy="no",                 # No evaluation
            save_strategy="steps",
            save_steps=50,
            save_total_limit=5,

            # ── Optimizer ──
            optim="adamw_8bit",
            seed=3407,
            report_to="none",
            disable_tqdm=False,
            output_dir="/checkpoints/outputs_v2",
        ),
    )

    # ── 6. Run training (resume from last checkpoint if available) ─────
    import glob
    checkpoint_dirs = glob.glob("/checkpoints/outputs_v2/checkpoint-*")
    resume_ckpt = None
    if checkpoint_dirs:
        resume_ckpt = max(checkpoint_dirs, key=lambda d: int(d.split("-")[-1]))
        print(f"🔄 Resuming from checkpoint: {resume_ckpt}")
    else:
        print("🔥 Starting LoRA-resumed training on entire dataset (2 epochs, lr=5e-5)...")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ── 7. Save final adapter ────────────────────────────────────────────
    print("💾 Saving final LoRA adapter + tokenizer...")
    model.save_pretrained("/checkpoints/lora_model_v2")
    tokenizer.save_pretrained("/checkpoints/lora_model_v2")
    print("✅ Saved to /checkpoints/lora_model_v2")


# ---------------------------------------------------------------------------
# Inference Function
# ---------------------------------------------------------------------------
@app.function(
    image=train_image,
    gpu="L40S",
    cpu=2.0,
    memory=32768,
    volumes={
        "/model_cache": model_cache_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[modal.Secret.from_dotenv()],
)
def run_inference_remote():
    import os
    import glob
    import torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    results = []

    # Prefer the v2 adapter, fall back to v1, then latest checkpoint
    model_path = "/checkpoints/lora_model_v2"
    if not os.path.exists(model_path):
        model_path = "/checkpoints/lora_model"
    if not os.path.exists(model_path):
        checkpoint_dirs = glob.glob("/checkpoints/outputs_v2/checkpoint-*")
        if checkpoint_dirs:
            model_path = max(checkpoint_dirs, key=os.path.getctime)
        else:
            model_path = "shinigamiRaj/IndicVedas-LoRA"

    results.append(f"Loading model for inference from: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=16384,          # 16k is fine for inference
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # MODE 1: Text Completion
    def complete_vedic_text(prompt, max_new_tokens=512):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.15,
        )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        return response.strip()

    # MODE 2: Q&A Chat
    chat_tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    def ask_vedagpt(query, max_new_tokens=512):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are VedaGPT, an expert scholar of the ancient Vedic scriptures "
                    "like RigVeda, SamaVeda, YajurVeda, AtharvaVeda, Charaka Samhita, "
                    "Sushruta Samhita, Rasa Jala Nidhi, IRJAY (International Research "
                    "Journal of Ayurveda and Yoga). "
                    "Answer questions accurately based on your knowledge of the Vedas, "
                    "Upanishads, Charaka Samhita, Sushruta Samhita, and other classical "
                    "Indian texts. "
                    "Maintain the style of writing as per the ancient Vedic texts where required."
                ),
            },
            {"role": "user", "content": query},
        ]
        inputs = chat_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.15,
        )
        response = tokenizer.decode(
            outputs[0][inputs.shape[-1]:],
            skip_special_tokens=True,
        )
        return response.strip()

    # ── Test Text Completion ──
    results.append("\n" + "=" * 60)
    results.append("TEXT COMPLETION TESTS")
    results.append("=" * 60)
    completion_prompts = [
        "[[ Collection: Rig Veda | Translator: Ralph T.H. Griffith | Book: 1 | Hymn: HYMN I | Title: Agni ]]\n\n",
        "[[ Collection: Atharva Veda | Translator: Ralph T.H. Griffith | Book: 1 | Hymn: HYMN I ]]\n\n",
        "[[ Collection: Charaka Samhita ]]\n\n",
        "[[ Collection: Sushruta Samhita ]]\n\n",
        "[[ Collection: Rasa Jala Nidhi ]]\n\n",
    ]
    for prompt in completion_prompts:
        results.append(f"\nPROMPT: {prompt[:80].strip()}...")
        results.append(f"COMPLETION: {complete_vedic_text(prompt)[:500]}")

    # ── Test Q&A Chat ──
    results.append("\n" + "=" * 60)
    results.append("Q&A CHAT TESTS")
    results.append("=" * 60)
    qa_queries = [
        "What are the four Vedas?",
        "Recite Hymn 1 from the Rig Veda.",
        "What does the Charaka Samhita say about digestion?",
        "Explain the concept of Rasa in Rasa Jala Nidhi.",
        "What surgical procedures does the Sushruta Samhita describe?",
    ]
    for q in qa_queries:
        results.append(f"\nQ: {q}")
        results.append(f"A: {ask_vedagpt(q)}")

    return "\n".join(results)


# ---------------------------------------------------------------------------
# Push & GGUF Export
# ---------------------------------------------------------------------------
@app.function(
    image=train_image,
    gpu="L40S",
    cpu=2.0,
    memory=32768,
    timeout=3 * 3600,
    volumes={
        "/model_cache": model_cache_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[modal.Secret.from_dotenv()],
)
def push_and_export_remote():
    import os
    import glob
    import torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    hf_token = os.environ.get("HF_TOKEN")

    model_path = "/checkpoints/lora_model_v2"
    if not os.path.exists(model_path):
        checkpoint_dirs = glob.glob("/checkpoints/outputs_v2/checkpoint-*")
        if checkpoint_dirs:
            model_path = max(checkpoint_dirs, key=os.path.getctime)
        else:
            raise FileNotFoundError(
                "No v2 checkpoints or saved model found. Run training first."
            )

    print(f"✅ Loading model from: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=16384,
        dtype=torch.bfloat16,
        load_in_4bit=False,     # full precision for merging
    )

    tokenizer = get_chat_template(tokenizer, chat_template="chatml")
    print("✅ Qwen ChatML template re-applied.")

    # 1. Push updated LoRA adapters
    print("📤 Pushing LoRA adapters to HF...")
    model.push_to_hub("shinigamiRaj/IndicVedas-LoRA", token=hf_token)
    tokenizer.push_to_hub("shinigamiRaj/IndicVedas-LoRA", token=hf_token)

    # 2. Push merged 16-bit model
    print("📤 Pushing merged 16-bit model to HF...")
    model.push_to_hub_merged(
        "shinigamiRaj/IndicVedas",
        tokenizer,
        save_method="merged_16bit",
        token=hf_token,
    )

    # 3. Export and push GGUF
    print("📤 Exporting and pushing GGUF...")
    model.push_to_hub_gguf(
        "shinigamiRaj/IndicVedas",
        tokenizer,
        quantization_method="q4_k_m",
        token=hf_token,
    )

    # 4. Local GGUF backup on volume
    print("💾 Saving GGUF locally...")
    model.save_pretrained_gguf(
        "/checkpoints/IndicVedas_GGUF_v2",
        tokenizer,
        quantization_method="q4_k_m",
    )
    print("🎉 All formats exported and uploaded!")


# ---------------------------------------------------------------------------
# Local Entrypoints
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def train():
    """Run the LoRA-resumed training job."""
    print("🚀 Launching LoRA-resumed training on Modal (14B, 2 epochs, entire dataset)...")
    call = train_remote.spawn()
    print(f"✅ Training job spawned! Function call ID: {call.object_id}")
    print("   The job will continue running on Modal even if this terminal closes.")
    print("   Monitor at: https://modal.com/apps/rajmanmauji")


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

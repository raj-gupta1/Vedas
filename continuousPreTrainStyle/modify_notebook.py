"""
Modify qlorafinetuneV2.ipynb for Continuous Pretraining (CLM).

Changes:
- Cell 9:  Load continueousPreTrainData.jsonl, show stats
- Cell 13: Remove chat template (not needed for CLM)
- Cell 15: Keep chunking (unchanged)
- Cell 17: Simple EOS formatting (no chat template)
- Cell 19: Load full data → chunk → split into train/val
- Cell 21: Updated SFTConfig: LR=2e-5, cosine, 3 epochs, eval
- Cell 27: Both text completion AND Q&A inference modes
"""

import json
import copy

NOTEBOOK_PATH = "qlorafinetunev2.ipynb"

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ─── Cell 9: Load full cleaned dataset and show stats ─────────────────────────
nb["cells"][9]["source"] = [
    "import json\n",
    "\n",
    "# Load the full cleaned dataset\n",
    "dataset_list = []\n",
    'with open("data/continueousPreTrainData.jsonl", "r", encoding="utf-8") as f:\n',
    "    for line in f:\n",
    "        if line.strip():\n",
    "            dataset_list.append(json.loads(line))\n",
    "\n",
    "# Token length check\n",
    'lengths = [len(tokenizer.encode(row["text"])) for row in dataset_list]\n',
    "\n",
    'print(f"Total rows: {len(dataset_list)}")\n',
    'print(f"Max token length: {max(lengths)}")\n',
    'print(f"Average token length: {sum(lengths) / len(lengths):.0f}")\n',
    'print(f"Rows > 4096 tokens: {sum(1 for l in lengths if l > 4096)}")\n',
]
nb["cells"][9]["outputs"] = []

# ─── Cell 13: Remove chat template — use base tokenizer for CLM ──────────────
nb["cells"][13]["source"] = [
    "# For continuous pretraining (CLM), we do NOT apply a chat template.\n",
    "# The model learns raw text directly. We just need to ensure the\n",
    "# tokenizer has proper EOS and PAD tokens set.\n",
    "\n",
    'print(f"EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")\n',
    'print(f"PAD token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")\n',
    "\n",
    "# Ensure pad token is set (some models don't have one)\n",
    "if tokenizer.pad_token is None:\n",
    "    tokenizer.pad_token = tokenizer.eos_token\n",
    '    print("Set pad_token = eos_token")\n',
    "\n",
    'print("✅ Tokenizer ready for continuous pretraining.")\n',
]
nb["cells"][13]["outputs"] = []

# ─── Cell 17: Simple CLM formatting (just append EOS) ────────────────────────
nb["cells"][17]["source"] = [
    "def formatting_pretrain_function(example):\n",
    '    """Format text for continuous pretraining (CLM).\n',
    "    \n",
    "    Simply append EOS token to the raw text so the model\n",
    "    learns document boundaries. No chat template needed.\n",
    '    """\n',
    '    text = example["text"].strip()\n',
    "    return {\"text\": text + tokenizer.eos_token}\n",
]
nb["cells"][17]["outputs"] = []

# ─── Cell 19: Load full → chunk → split into train/val ───────────────────────
nb["cells"][19]["source"] = [
    "from datasets import Dataset\n",
    "import random\n",
    "\n",
    "# 1. Chunk the FULL dataset first\n",
    'print("Chunking full dataset (4096 tokens, 200 overlap)...")\n',
    "chunked_all = chunk_dataset_tokenized(dataset_list, tokenizer)\n",
    "\n",
    "# 2. Shuffle and split into train/val (90/10)\n",
    "random.seed(42)\n",
    "random.shuffle(chunked_all)\n",
    "val_size = max(1, int(len(chunked_all) * 0.1))\n",
    "chunked_val = chunked_all[:val_size]\n",
    "chunked_train = chunked_all[val_size:]\n",
    "\n",
    'print(f"After chunking & split: train={len(chunked_train)}, val={len(chunked_val)}")\n',
    "\n",
    "# 3. Convert to HuggingFace Datasets and format for CLM (append EOS)\n",
    'print("Formatting for continuous pretraining (appending EOS)...")\n',
    "dataset_train = Dataset.from_list(chunked_train)\n",
    "dataset_train = dataset_train.map(formatting_pretrain_function, batched=False)\n",
    "\n",
    "dataset_val = Dataset.from_list(chunked_val)\n",
    "dataset_val = dataset_val.map(formatting_pretrain_function, batched=False)\n",
    "\n",
    'print(f"Training examples:   {len(dataset_train)}")\n',
    'print(f"Validation examples: {len(dataset_val)}")\n',
    "\n",
    "# 4. Verify token lengths\n",
    "sample_lengths = [len(tokenizer.encode(dataset_train[i]['text'])) for i in range(min(100, len(dataset_train)))]\n",
    'print(f"\\n--- Token Length Check (first 100 train rows) ---")\n',
    'print(f"Max: {max(sample_lengths)}, Avg: {sum(sample_lengths)/len(sample_lengths):.0f}")\n',
    "\n",
    "# 5. Preview what the model sees during training\n",
    'print("\\n--- WHAT THE MODEL SEES DURING TRAINING ---")\n',
    "print(dataset_train[0]['text'][:600])\n",
]
nb["cells"][19]["outputs"] = []

# ─── Cell 21: Updated SFTConfig for CLM ───────────────────────────────────────
nb["cells"][21]["source"] = [
    "import torch\n",
    "from trl import SFTTrainer, SFTConfig\n",
    "\n",
    "use_bf16 = torch.cuda.is_bf16_supported()\n",
    "\n",
    "trainer = SFTTrainer(\n",
    "    model = model,\n",
    "    train_dataset = dataset_train,\n",
    "    eval_dataset = dataset_val,              # Validation set for monitoring\n",
    "    processing_class = tokenizer,\n",
    "    args = SFTConfig(\n",
    '        dataset_text_field = "text",\n',
    "        max_seq_length = max_seq_length,\n",
    "        dataset_num_proc = 4,\n",
    "        packing = True,                       # Pack short texts together for efficiency\n",
    "        per_device_train_batch_size = 4,\n",
    "        gradient_accumulation_steps = 2,\n",
    "        warmup_steps = 50,                    # More warmup for stability\n",
    "        num_train_epochs = 3,                 # 3 epochs for deeper domain adaptation\n",
    "        learning_rate = 2e-5,                 # Lower LR for 14B model (was 1e-4)\n",
    '        lr_scheduler_type = "cosine",         # Cosine decay for smooth convergence\n',
    "        weight_decay = 0.01,                  # Regularization to prevent overfitting\n",
    "        fp16 = not use_bf16,\n",
    "        bf16 = use_bf16,\n",
    "        logging_steps = 5,\n",
    '        eval_strategy = "steps",              # Evaluate during training\n',
    "        eval_steps = 100,                     # Eval every 100 steps\n",
    '        optim = "adamw_8bit",\n',
    "        seed = 3407,\n",
    '        report_to = "none",\n',
    "\n",
    "        # Checkpoint saving\n",
    '        output_dir = "outputs",\n',
    '        save_strategy = "steps",\n',
    "        save_steps = 100,\n",
    "        save_total_limit = 3,\n",
    "        load_best_model_at_end = True,        # Auto-load best checkpoint\n",
    '        metric_for_best_model = "eval_loss",  # Best = lowest eval loss\n',
    "    ),\n",
    ")\n",
]
nb["cells"][21]["outputs"] = []

# ─── Cell 27: Both text completion AND Q&A inference ──────────────────────────
nb["cells"][27]["source"] = [
    "from unsloth.chat_templates import get_chat_template\n",
    "FastLanguageModel.for_inference(model)\n",
    "\n",
    "# ═══════════════════════════════════════════════════════════════\n",
    "# MODE 1: Text Completion (CLM style)\n",
    "# Feed a structured header and the model continues the text.\n",
    "# ═══════════════════════════════════════════════════════════════\n",
    "def complete_vedic_text(prompt, max_new_tokens=256):\n",
    '    """Complete a Vedic text prompt."""\n',
    '    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")\n',
    "    outputs = model.generate(\n",
    "        **inputs,\n",
    "        max_new_tokens = max_new_tokens,\n",
    "        use_cache = True,\n",
    "        temperature = 0.7,\n",
    "        do_sample = True,\n",
    "        top_p = 0.9,\n",
    "        repetition_penalty = 1.15,\n",
    "    )\n",
    "    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)\n",
    "    return response.strip()\n",
    "\n",
    "# ═══════════════════════════════════════════════════════════════\n",
    "# MODE 2: Q&A Chat (uses base Qwen2.5-Instruct chat abilities)\n",
    "# The base model still has its chat capabilities intact since\n",
    "# we only trained a LoRA adapter on top.\n",
    "# ═══════════════════════════════════════════════════════════════\n",
    "chat_tokenizer = get_chat_template(tokenizer, chat_template='chatml')\n",
    "\n",
    "def ask_vedagpt(query, max_new_tokens=256):\n",
    '    """Chat with VedaGPT using the base model chat abilities."""\n',
    "    messages = [\n",
    "        {\n",
    '            "role": "system",\n',
    '            "content": "You are VedaGPT, an expert scholar of the ancient Vedic scriptures, Ayurveda, and Yoga. "\n',
    '                       "Answer questions accurately based on your knowledge of the Vedas."\n',
    "        },\n",
    '        {"role": "user", "content": query},\n',
    "    ]\n",
    "\n",
    "    inputs = chat_tokenizer.apply_chat_template(\n",
    "        messages,\n",
    "        tokenize = True,\n",
    "        add_generation_prompt = True,\n",
    '        return_tensors = "pt",\n',
    '    ).to("cuda")\n',
    "\n",
    "    outputs = model.generate(\n",
    "        input_ids = inputs,\n",
    "        max_new_tokens = max_new_tokens,\n",
    "        use_cache = True,\n",
    "        temperature = 0.7,\n",
    "        do_sample = True,\n",
    "        top_p = 0.9,\n",
    "        repetition_penalty = 1.15,\n",
    "    )\n",
    "    response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)\n",
    "    return response.strip()\n",
    "\n",
    "# ─── Test Text Completion ─────────────────────────────────────\n",
    'print("\\n" + "="*60)\n',
    'print("TEXT COMPLETION TESTS")\n',
    'print("="*60)\n',
    "completion_prompts = [\n",
    '    "[[ Collection: Rig Veda | Translator: Ralph T.H. Griffith | Book: 1 | Hymn: HYMN I | Title: Agni ]]\\n\\n",\n',
    '    "[[ Collection: Atharva Veda | Translator: Ralph T.H. Griffith | Book: 1 | Hymn: HYMN I ]]\\n\\n",\n',
    '    "[[ Collection: Charaka Samhita ]]\\n\\n",\n',
    "]\n",
    "for prompt in completion_prompts:\n",
    '    print(f"\\nPROMPT: {prompt[:80].strip()}...")\n',
    '    print(f"COMPLETION: {complete_vedic_text(prompt)[:300]}")\n',
    "\n",
    "# ─── Test Q&A Chat ────────────────────────────────────────────\n",
    'print("\\n" + "="*60)\n',
    'print("Q&A CHAT TESTS")\n',
    'print("="*60)\n',
    "qa_queries = [\n",
    '    "What are the four Vedas?",\n',
    '    "Recite Hymn 1 from the Rig Veda.",\n',
    '    "What does the Charaka Samhita say about digestion?",\n',
    "]\n",
    "for q in qa_queries:\n",
    '    print(f"\\nQ: {q}")\n',
    '    print(f"A: {ask_vedagpt(q)}")\n',
]
nb["cells"][27]["outputs"] = []

# ─── Save modified notebook ───────────────────────────────────────────────────
with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("✅ Notebook modified successfully!")
print()
print("Changes:")
print("  Cell  9: Load full continueousPreTrainData.jsonl")
print("  Cell 13: Removed chat template (CLM mode)")
print("  Cell 17: Simple EOS formatting")
print("  Cell 19: Chunk full data FIRST → then split 90/10 into train/val")
print("  Cell 21: LR=2e-5, cosine, 3 epochs, eval every 100 steps")
print("  Cell 27: Both text completion AND Q&A chat inference")

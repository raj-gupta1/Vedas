import os
import json
from datasets import load_dataset

# Script base directory for absolute relative path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data")
RAW_PRETRAIN_FILE = os.path.join(OUTPUT_DIR, "continueousPreTrainData.jsonl")
TRAIN_FILE = os.path.join(OUTPUT_DIR, "train_split.jsonl")
TEST_FILE = os.path.join(OUTPUT_DIR, "test_split.jsonl")
VAL_FILE = os.path.join(OUTPUT_DIR, "val_split.jsonl")

def main():
    print("🧹 Starting Continuous Pre-Training Data Split Preparation (80/10/10 Train/Val/Test)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Merge step: Automatically find all files ending in _pretrain.jsonl in OUTPUT_DIR and merge them
    print("🔗 Checking for *_pretrain.jsonl files in data directory to merge...")
    pretrain_files = sorted([
        os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR)
        if f.endswith("_pretrain.jsonl") and os.path.join(OUTPUT_DIR, f) != RAW_PRETRAIN_FILE
    ])
    
    if not pretrain_files:
        print("⚠️ No *_pretrain.jsonl files found in data directory. Skipping merge.")
    else:
        print(f"🔗 Merging {len(pretrain_files)} files into {RAW_PRETRAIN_FILE}...")
        total_records_merged = 0
        with open(RAW_PRETRAIN_FILE, "w", encoding="utf-8") as out_f:
            for file_path in pretrain_files:
                file_records = 0
                with open(file_path, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        if line.strip():
                            out_f.write(line)
                            file_records += 1
                print(f"   📄 Merged {file_records} records from {os.path.basename(file_path)}")
                total_records_merged += file_records
        print(f"✅ Merged total of {total_records_merged} records into: {RAW_PRETRAIN_FILE}")
    
    if not os.path.exists(RAW_PRETRAIN_FILE):
        raise FileNotFoundError(f"Could not find continuous pre-train data file at: {RAW_PRETRAIN_FILE}")
    
    print(f"📖 Reading raw pre-train data from: {RAW_PRETRAIN_FILE}")
    import random
    raw_records = []
    with open(RAW_PRETRAIN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_records.append(json.loads(line))
    print(f"📝 Loaded {len(raw_records)} total raw records.")
    
    # Shuffle and split into 80% train, 10% val, 10% test
    print("🎲 Shuffling and splitting dataset into 80% Train, 10% Val, 10% Test...")
    random.seed(42)
    random.shuffle(raw_records)
    
    num_records = len(raw_records)
    if num_records < 10:
        raise ValueError(f"Dataset has only {num_records} records, which is too small to split!")
        
    train_size = int(num_records * 0.8)
    val_size = int(num_records * 0.1)
    test_size = num_records - train_size - val_size
    
    print(f"📊 Split sizes: Train={train_size}, Val={val_size}, Test={test_size}")
    
    train_dataset = raw_records[:train_size]
    val_dataset = raw_records[train_size:train_size + val_size]
    test_dataset = raw_records[train_size + val_size:]
    
    # Save splits
    for name, ds, path in [
        ("Train", train_dataset, TRAIN_FILE), 
        ("Test", test_dataset, TEST_FILE), 
        ("Val", val_dataset, VAL_FILE)
    ]:
        with open(path, "w", encoding="utf-8") as out_f:
            for row in ds:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"✅ Saved {len(ds)} {name} split records to: {path}")

if __name__ == "__main__":
    main()

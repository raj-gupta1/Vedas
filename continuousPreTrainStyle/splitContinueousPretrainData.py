import os
import json
from datasets import load_dataset

# Script base directory for absolute relative path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration
RAW_PRETRAIN_FILE = os.path.join(SCRIPT_DIR, "continueousPreTrainData.jsonl")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data")
TRAIN_FILE = os.path.join(OUTPUT_DIR, "train_split.jsonl")
TEST_FILE = os.path.join(OUTPUT_DIR, "test_split.jsonl")
VAL_FILE = os.path.join(OUTPUT_DIR, "val_split.jsonl")

def main():
    print("🧹 Starting Continuous Pre-Training Data Split Preparation (50 Train/Test/Val)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(RAW_PRETRAIN_FILE):
        raise FileNotFoundError(f"Could not find continuous pre-train data file at: {RAW_PRETRAIN_FILE}")
    
    print(f"📖 Reading raw pre-train data from: {RAW_PRETRAIN_FILE}")
    raw_dataset = load_dataset("json", data_files=RAW_PRETRAIN_FILE)["train"]
    print(f"📝 Loaded {len(raw_dataset)} total raw records.")
    
    # Shuffle and select 50 random unique records for each split
    print("🎲 Shuffling and selecting exactly 50 random unique samples each for train, test, and validation...")
    shuffled_dataset = raw_dataset.shuffle(seed=42)
    
    if len(shuffled_dataset) < 150:
        raise ValueError(f"Dataset has only {len(shuffled_dataset)} records, which is less than the required 150 unique records (50 for train, 50 for test, 50 for val)!")
        
    train_dataset = shuffled_dataset.select(range(0, 50))
    test_dataset = shuffled_dataset.select(range(50, 100))
    val_dataset = shuffled_dataset.select(range(100, 150))
    
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

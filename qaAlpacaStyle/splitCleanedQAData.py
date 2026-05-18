import os
import json

# Configuration
RAW_QA_FILE = "cleaned_QAData.jsonl"
OUTPUT_DIR = "data"
TRAIN_FILE = os.path.join(OUTPUT_DIR, "train_split.jsonl")
TEST_FILE = os.path.join(OUTPUT_DIR, "test_split.jsonl")
VAL_FILE = os.path.join(OUTPUT_DIR, "val_split.jsonl")

def main():
    print("🧹 Starting QA Data Split Preparation...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if raw QA data is in the directory above or in the current directory
    raw_path = RAW_QA_FILE
    if not os.path.exists(raw_path):
        raw_path = "cleaned_QAData.jsonl"
        if not os.path.exists(raw_path):
            # Try absolute path
            raw_path = "/Users/raj/PycharmProjects/VedaGPT/cleaned_QAData.jsonl"
            if not os.path.exists(raw_path):
                raise FileNotFoundError("Could not find cleaned_QAData.jsonl in root or parent directories!")
    
    print(f"📖 Reading raw QA data from: {raw_path}")
    with open(raw_path, "r", encoding="utf-8") as f:
        records = [json.loads(line.strip()) for line in f]
        
    print(f"📝 Total raw QA records loaded: {len(records)}")
    
    # Split evenly: 20 train, 20 test, 20 validation as configured earlier
    train_records = records[:20]
    test_records = records[20:40]
    val_records = records[40:60]
    
    # Write splits
    for name, split_records, path in [("Train", train_records, TRAIN_FILE), 
                                     ("Test", test_records, TEST_FILE), 
                                     ("Val", val_records, VAL_FILE)]:
        with open(path, "w", encoding="utf-8") as out_f:
            for rec in split_records:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"✅ Saved {len(split_records)} {name} split records to: {path}")

if __name__ == "__main__":
    main()

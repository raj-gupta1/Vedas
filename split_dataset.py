import json
import random
import os

# Configuration
INPUT_FILE = "cleaned_train.jsonl"
TRAIN_OUTPUT = "data/train_split.jsonl"
TEST_OUTPUT = "data/test_split.jsonl"
SAMPLE_SIZE = 20

def split_dataset():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Source file '{INPUT_FILE}' not found!")
        return

    # Ensure output directory exists
    os.makedirs("data", exist_ok=True)

    print(f"Reading dataset from: {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        records = [json.loads(line.strip()) for line in f if line.strip()]

    total_records = len(records)
    print(f"📊 Total records found: {total_records}")

    if total_records < (SAMPLE_SIZE * 2):
        print(f"⚠️ Warning: Dataset has only {total_records} records, which is less than the requested {SAMPLE_SIZE * 2} split.")
        # Adjust sample size to be at most half of the dataset
        split_size = total_records // 2
        print(f"🔄 Adjusting splits to {split_size} training and {split_size} testing records.")
    else:
        split_size = SAMPLE_SIZE

    # Randomly shuffle records
    print("🎲 Shuffling records randomly...")
    random.shuffle(records)

    # Slice the random subsets
    train_records = records[:split_size]
    test_records = records[split_size : split_size * 2]

    # Save training subset
    with open(TRAIN_OUTPUT, "w", encoding="utf-8") as f:
        for record in train_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Save testing subset
    with open(TEST_OUTPUT, "w", encoding="utf-8") as f:
        for record in test_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✅ Splits successfully created!")
    print(f"📝 Training data saved to: {TRAIN_OUTPUT} ({len(train_records)} records)")
    print(f"📝 Testing data saved to: {TEST_OUTPUT} ({len(test_records)} records)")

if __name__ == "__main__":
    split_dataset()

import json
import os

def clean_train_dataset(file_path="train.jsonl"):
    if not os.path.exists(file_path):
        print(f"❌ File {file_path} not found.")
        return

    print(f"🧹 Reading and cleaning {file_path}...")
    cleaned_count = 0
    cleaned_records = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Check if this is a valid instruction-output record
            if "instruction" in data and "output" in data:
                clean_record = {
                    "instruction": data["instruction"],
                    "output": data["output"]
                }
                cleaned_records.append(clean_record)
                cleaned_count += 1

    output_path = "cleaned_train.jsonl"
    # Write the clean records to a separate file (do not overwrite the original file!)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in cleaned_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"✅ Success! Cleaned and saved {cleaned_count} lines to '{output_path}'.")
    print(f"✨ Kept the original '{file_path}' completely untouched with all of its rich metadata!")

if __name__ == "__main__":
    clean_train_dataset()

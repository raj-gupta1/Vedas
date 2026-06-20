import os
import json
from huggingface_hub import hf_hub_download, HfApi
from dotenv import load_dotenv

load_dotenv("/Users/raj/PycharmProjects/VedaGPT/.env")

hf_token = os.getenv("HF_TOKEN")
api = HfApi(token=hf_token)

repo_id = "shinigamiRaj/IndicVedas"
base_model = "unsloth/Qwen2.5-14B-Instruct"

print(f"Downloading correct config.json from {base_model}...")
base_config_path = hf_hub_download(repo_id=base_model, filename="config.json")

with open(base_config_path, "r") as f:
    config_data = json.load(f)

# The user's repo is qwen2, ensure architectures is set
if "architectures" not in config_data:
    config_data["architectures"] = ["Qwen2ForCausalLM"]

# Save locally to upload
temp_config_path = "temp_config.json"
with open(temp_config_path, "w") as f:
    json.dump(config_data, f, indent=2)

print(f"Uploading fixed config.json to {repo_id}...")
api.upload_file(
    path_or_fileobj=temp_config_path,
    path_in_repo="config.json",
    repo_id=repo_id,
    commit_message="Fix config.json architectures for vLLM compatibility"
)

os.remove(temp_config_path)
print("✅ Successfully updated config.json!")

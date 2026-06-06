"""
VedaGPT Interactive Inference on Modal (vLLM-powered)
=====================================================

Uses vLLM for blazing-fast inference instead of Unsloth (which is
designed for training, not serving). vLLM provides:
  - Continuous batching & PagedAttention for high throughput
  - ~5-10x faster token generation vs naive HF generate()

Usage:
    source continuousPreTrainStyle/vedaFineTune/bin/activate

    # Interactive chat (recommended):
    modal run -q continuousPreTrainStyle/modalinference.py

    # Single prompt:
    modal run continuousPreTrainStyle/modalinference.py --prompt "What are the four Vedas?"

    # Run default test suite:
    modal run continuousPreTrainStyle/modalinference.py --test

    # Text completion mode (instead of Q&A chat):
    modal run continuousPreTrainStyle/modalinference.py --mode completion
"""

import os
import sys
import time

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import modal

app = modal.App("vedagpt-vllm-inference")

# Persistent volumes (same as training script)
model_cache_volume = modal.Volume.from_name("unsloth-model-cache", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("unsloth-checkpoints", create_if_missing=True)

# ---------------------------------------------------------------------------
# Container Image — lightweight, vLLM-focused
# ---------------------------------------------------------------------------
HF_MODEL_ID = "shinigamiRaj/IndicVedas"

inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.8.5.post1",
        "transformers==4.51.3",
        "huggingface_hub>=0.23.0",
    )
    .env({
        "HF_HOME": "/model_cache",
        "VLLM_USE_V1": "0",  # Use stable v0 engine (v1 multiprocessing fails in Modal)
    })
)


# ---------------------------------------------------------------------------
# vLLM Inference Class (keeps model warm in GPU memory)
# ---------------------------------------------------------------------------
@app.cls(
    image=inference_image,
    gpu="L40S",
    cpu=2.0,
    memory=32768,
    timeout=600,
    volumes={
        "/model_cache": model_cache_volume,
        "/checkpoints": checkpoint_volume,
    },
)
class VedaGPTInference:

    @modal.enter()
    def load_model(self):
        """Load model once when the container starts. Stays warm for follow-up calls."""
        import os
        from vllm import LLM

        # Prefer the merged model on HuggingFace, fall back to local volume
        local_merged = "/checkpoints/merged_model"
        if os.path.isdir(local_merged) and os.listdir(local_merged):
            model_source = local_merged
            print(f"📂 Loading model from local volume: {model_source}")
        else:
            model_source = HF_MODEL_ID
            print(f"📂 Loading model from HuggingFace: {model_source}")

        t0 = time.time()
        self.llm = LLM(
            model=model_source,
            max_model_len=4096,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            enforce_eager=True,
            disable_log_stats=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        elapsed = time.time() - t0
        print(f"✅ vLLM engine ready in {elapsed:.1f}s")

    @modal.method()
    def generate(self, prompt: str, mode: str = "chat", max_new_tokens: int = 1000) -> str:
        """Generate a response using vLLM."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=0.2,
            top_p=0.9,
            max_tokens=max_new_tokens,
            repetition_penalty=1.15,
            stop=["<|im_end|>", "<|endoftext|>"],
        )

        if mode == "chat":
            messages = [
                {
                    "role": "system",
                    "content": (
                    "You are VedaGPT, an expert scholar of the ancient Vedic scriptures like RigVeda, SamaVeda, YajurVeda, AtharvaVeda, Charaka Samhita, Sushruta Samhita, Rasa Jala Nidhi, IRJAY (International Research Journal of Ayurveda and Yoga)"
                           "Answer questions accurately based on your knowledge of the Vedas, Upanishads, Charaka Samhita, Sushruta Samhita, and other classical Indian texts."
                            "Maintain the style of writing as per the ancient Vedic texts where required."
                ),
                },
                {"role": "user", "content": prompt},
            ]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            # Raw text completion
            formatted = prompt

        outputs = self.llm.generate([formatted], sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text.strip()


# ---------------------------------------------------------------------------
# CLI Helpers
# ---------------------------------------------------------------------------
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


def print_banner():
    print(f"""
{BOLD}{CYAN}╔════════════════════════════════════════════════════════════╗
║          🙏  VedaGPT — Modal vLLM Inference  🙏          ║
╚════════════════════════════════════════════════════════════╝{RESET}
""")


def print_separator():
    print(f"{DIM}{CYAN}────────────────────────────────────────────────────────────{RESET}")


def run_interactive(inference: VedaGPTInference, mode: str):
    """Interactive chat loop with clear prompts."""
    print(f"  {BOLD}Interactive Session{RESET}")
    print(f"  Type your question and press Enter.")
    print(f"  Type {YELLOW}'exit'{RESET} or {YELLOW}'quit'{RESET} to stop.\n")

    # Warm the container BEFORE asking for input so Modal's loading spinner doesn't eat stdin
    print(f"  {DIM}⏳ Warming up the remote vLLM engine (may take ~1-2 min)...{RESET}")
    try:
        inference.generate.remote("warmup", mode="completion", max_new_tokens=1000)
        print(f"  {GREEN}✅ Engine ready!{RESET}\n")
    except Exception as e:
        print(f"  {RED}❌ Engine startup failed: {e}{RESET}\n")
        return

    print_separator()

    while True:
        try:
            sys.stdout.flush()
            user_input = input(f"\n  {BOLD}{YELLOW}You ▸ {RESET}").strip()

            if user_input.lower() in ("exit", "quit", "q"):
                print(f"\n  {CYAN}🙏 Dhanyavad! Exiting session.{RESET}\n")
                break
            if not user_input:
                continue

            print(f"\n  {DIM}⏳ Generating response...{RESET}")
            t0 = time.time()
            response = inference.generate.remote(user_input, mode=mode)
            elapsed = time.time() - t0

            print(f"\n  {BOLD}{GREEN}VedaGPT ▸{RESET}")
            # Indent each line of the response for readability
            for line in response.split("\n"):
                print(f"  {line}")
            print(f"\n  {DIM}({elapsed:.1f}s){RESET}")
            print_separator()

        except KeyboardInterrupt:
            print(f"\n\n  {CYAN}🙏 Dhanyavad! Exiting session.{RESET}\n")
            break
        except Exception as e:
            print(f"\n  {RED}❌ Error: {e}{RESET}\n")


def run_single_prompt(inference: VedaGPTInference, prompt: str, mode: str):
    """Run a single prompt and print the result."""
    print(f"  {BOLD}Single Prompt Mode{RESET}\n")
    print(f"  {BOLD}{YELLOW}You ▸{RESET} {prompt}\n")
    print(f"  {DIM}⏳ Generating response...{RESET}")
    t0 = time.time()
    response = inference.generate.remote(prompt, mode=mode)
    elapsed = time.time() - t0
    print(f"\n  {BOLD}{GREEN}VedaGPT ▸{RESET}")
    for line in response.split("\n"):
        print(f"  {line}")
    print(f"\n  {DIM}({elapsed:.1f}s){RESET}")
    print_separator()


def run_test_suite(inference: VedaGPTInference, mode: str):
    """Run a predefined suite of test questions."""
    test_questions = [
        # Knowledge breadth
        "What are the four main Vedas and what is the primary focus of each?",
        # Specific retrieval
        "Recite or describe Hymn I of the Rig Veda dedicated to Agni.",
        # Ayurvedic knowledge
        "What does the Charaka Samhita say about the importance of digestion (Agni) in health?",
        # Philosophical depth
        "Explain the concept of Purusha Sukta from the Rig Veda.",
        # Cross-text synthesis
        "How do the Yoga Sutras of Patanjali relate to the philosophy described in the Upanishads?",
        # Text completion test
        "[[ Collection: Rig Veda | Translator: Ralph T.H. Griffith | Book: 1 | Hymn: HYMN I | Title: Agni ]]\n\n",
    ]

    print(f"  {BOLD}Running {len(test_questions)} Test Questions{RESET}\n")

    for i, question in enumerate(test_questions, 1):
        # Use completion mode for the last test (text completion)
        q_mode = "completion" if i == len(test_questions) else mode

        display_q = question[:100].strip() + ("..." if len(question) > 100 else "")
        print(f"  {BOLD}{CYAN}Q{i}.{RESET} {display_q}")
        print(f"  {DIM}⏳ Generating...{RESET}")

        t0 = time.time()
        response = inference.generate.remote(question, mode=q_mode, max_new_tokens=1000)
        elapsed = time.time() - t0

        print(f"\n  {BOLD}{GREEN}A{i}.{RESET}")
        for line in response.split("\n"):
            print(f"  {line}")
        print(f"\n  {DIM}({elapsed:.1f}s){RESET}")
        print_separator()
        print()


# ---------------------------------------------------------------------------
# Local Entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    prompt: str = None,
    interactive: bool = False,
    test: bool = False,
    mode: str = "chat",
):
    """
    VedaGPT inference powered by vLLM on Modal.

    Args:
        prompt:      A single question/prompt to send.
        interactive: Start an interactive chat loop.
        test:        Run the default test suite.
        mode:        'chat' for Q&A or 'completion' for text completion.
    """
    print_banner()

    inference = VedaGPTInference()

    if prompt:
        run_single_prompt(inference, prompt, mode)
    elif test:
        run_test_suite(inference, mode)
    else:
        # Default to interactive
        run_interactive(inference, mode)

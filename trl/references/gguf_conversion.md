# GGUF Conversion Guide

After training models with TRL on Hugging Face Jobs, convert them to **GGUF format** for use with llama.cpp, Ollama, LM Studio, and other local inference tools.

**This guide provides production-ready, tested code.** All required dependencies are included in the examples below. No additional troubleshooting should be needed when following the templates exactly.

## What is GGUF?

**GGUF** (GPT-Generated Unified Format):
- Optimized format for CPU/GPU inference with llama.cpp
- Supports quantization (4-bit, 5-bit, 8-bit) to reduce model size
- Compatible with: Ollama, LM Studio, Jan, GPT4All, llama.cpp
- Typically 2-8GB for 7B models (vs 14GB unquantized)

## When to Convert to GGUF

**Convert when:**
- Running models locally with Ollama or LM Studio
- Using CPU-optimized inference
- Reducing model size with quantization
- Deploying to edge devices
- Sharing models for local-first use

## Conversion Process

**The conversion requires:**
1. **Merge LoRA adapter** with base model (if using PEFT)
2. **Convert to GGUF** format using llama.cpp
3. **Quantize** to different bit depths (optional but recommended)
4. **Upload** GGUF files to Hub

## GGUF Conversion Script Template

See `scripts/convert_to_gguf.py` for a complete, production-ready conversion script.

**Quick conversion job:**

```python
hf_jobs("uv", {
    "script": """
# /// script
# dependencies = [
#     "transformers>=4.36.0",
#     "peft>=0.7.0",
#     "torch>=2.0.0",
#     "huggingface_hub>=0.20.0",
#     "sentencepiece>=0.1.99",
#     "protobuf>=3.20.0",
#     "numpy",
#     "gguf",
# ]
# ///

import os
import torch
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi

# Configuration from environment
ADAPTER_MODEL = os.environ.get("ADAPTER_MODEL", "username/my-model")
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B")
OUTPUT_REPO = os.environ.get("OUTPUT_REPO", "username/my-model-gguf")

print("🔄 Converting to GGUF...")

# Step 1: Load and merge
print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print("Loading adapter...")
model = PeftModel.from_pretrained(base, ADAPTER_MODEL)

print("Merging...")
merged = model.merge_and_unload()

# Save merged model
merged_dir = "/tmp/merged"
merged.save_pretrained(merged_dir, safe_serialization=True)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_MODEL)
tokenizer.save_pretrained(merged_dir)

# Step 2: Install build tools and clone llama.cpp
print("Setting up llama.cpp...")
subprocess.run(["apt-get", "update", "-qq"], check=True, capture_output=True)
subprocess.run(["apt-get", "install", "-y", "-qq", "build-essential", "cmake"], check=True, capture_output=True)

subprocess.run([
    "git", "clone",
    "https://github.com/ggerganov/llama.cpp.git",
    "/tmp/llama.cpp"
], check=True)

subprocess.run([
    "pip", "install", "-r",
    "/tmp/llama.cpp/requirements.txt"
], check=True)

# Convert to GGUF
print("Converting to GGUF...")
subprocess.run([
    "python", "/tmp/llama.cpp/convert_hf_to_gguf.py",
    merged_dir,
    "--outfile", "/tmp/model-f16.gguf",
    "--outtype", "f16"
], check=True)

# Step 3: Build quantization tool with CMake
print("Building quantization tool...")
os.makedirs("/tmp/llama.cpp/build", exist_ok=True)

subprocess.run([
    "cmake", "-B", "/tmp/llama.cpp/build", "-S", "/tmp/llama.cpp",
    "-DGGML_CUDA=OFF"
], check=True)

subprocess.run([
    "cmake", "--build", "/tmp/llama.cpp/build",
    "--target", "llama-quantize", "-j", "4"
], check=True)

quantize = "/tmp/llama.cpp/build/bin/llama-quantize"
quants = ["Q4_K_M", "Q5_K_M", "Q8_0"]

for q in quants:
    print(f"Creating {q} quantization...")
    subprocess.run([
        quantize,
        "/tmp/model-f16.gguf",
        f"/tmp/model-{q.lower()}.gguf",
        q
    ], check=True)

# Step 4: Upload
print("Uploading to Hub...")
api = HfApi()
api.create_repo(OUTPUT_REPO, repo_type="model", exist_ok=True)

for q in ["f16"] + [q.lower() for q in quants]:
    api.upload_file(
        path_or_fileobj=f"/tmp/model-{q}.gguf",
        path_in_repo=f"model-{q}.gguf",
        repo_id=OUTPUT_REPO
    )

print(f"✅ Done! Models at: https://huggingface.co/{OUTPUT_REPO}")
""",
    "flavor": "a10g-large",
    "timeout": "45m",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"},
    "env": {
        "ADAPTER_MODEL": "username/my-finetuned-model",
        "BASE_MODEL": "Qwen/Qwen2.5-0.5B",
        "OUTPUT_REPO": "username/my-model-gguf"
    }
})
```

## Quantization Options

Common quantization formats (from smallest to largest):

| Format | Size | Quality | Use Case |
|--------|------|---------|----------|
| **Q4_K_M** | ~300MB | Good | **Recommended** - best balance of size/quality |
| **Q5_K_M** | ~350MB | Better | Higher quality, slightly larger |
| **Q8_0** | ~500MB | Very High | Near-original quality |
| **F16** | ~1GB | Original | Full precision, largest file |

**Recommendation:** Create Q4_K_M, Q5_K_M, and Q8_0 versions to give users options.

## Hardware Requirements

**For conversion:**
- Small models (<1B): CPU-basic works, but slow
- Medium models (1-7B): a10g-large recommended
- Large models (7B+): a10g-large or a100-large

**Time estimates:**
- 0.5B model: ~15-25 minutes on A10G
- 3B model: ~30-45 minutes on A10G
- 7B model: ~45-60 minutes on A10G

## Using GGUF Models

**GGUF models work on both CPU and GPU.** They're optimized for CPU inference but can also leverage GPU acceleration when available.

**With Ollama (auto-detects GPU):**
```bash
# Download GGUF
huggingface-cli download username/my-model-gguf model-q4_k_m.gguf

# Create Modelfile
echo "FROM ./model-q4_k_m.gguf" > Modelfile

# Create and run (uses GPU automatically if available)
ollama create my-model -f Modelfile
ollama run my-model
```

**With llama.cpp:**
```bash
# CPU only
./llama-cli -m model-q4_k_m.gguf -p "Your prompt"

# With GPU acceleration (offload 32 layers to GPU)
./llama-cli -m model-q4_k_m.gguf -ngl 32 -p "Your prompt"
```

**With LM Studio:**
1. Download the `.gguf` file
2. Import into LM Studio
3. Start chatting

## Best Practices

1. **Always create multiple quantizations** - Give users choice of size/quality
2. **Include README** - Document which quantization to use for what purpose
3. **Test the GGUF** - Run a quick inference test before uploading
4. **Use A10G GPU** - Much faster than CPU for loading/merging large models
5. **Clean up temp files** - Conversion creates large intermediate files

## Common Issues

**Out of memory during merge:**
- Use larger GPU (a10g-large or a100-large)
- Load with `device_map="auto"` for automatic device placement
- Use `dtype=torch.float16` or `torch.bfloat16` instead of float32

**Conversion fails with architecture error:**
- Ensure llama.cpp supports the model architecture
- Check that model uses standard architecture (Qwen, Llama, Mistral, etc.)
- Some newer models require latest llama.cpp from main branch
- Check llama.cpp issues/docs for model support

**GGUF file doesn't work with llama.cpp:**
- Verify llama.cpp version compatibility
- Download latest llama.cpp: `git clone https://github.com/ggerganov/llama.cpp.git`
- Rebuild llama.cpp after updating: `make clean && make`

**Quantization fails:**
- Ensure the `llama-quantize` tool was built: `make llama-quantize`
- Check that FP16 GGUF was created successfully before quantizing
- Some quantization types require specific llama.cpp versions

**Upload fails or times out:**
- Large models (>2GB) may need longer timeout
- Use `api.upload_file()` with `commit_message` for better tracking
- Consider uploading quantized versions separately

**See:** `scripts/convert_to_gguf.py` for complete, production-ready conversion script with all dependencies included.

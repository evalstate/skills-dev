# UV Scripts Guide for TRL Training

UV scripts are self-contained Python scripts with inline dependency declarations (PEP 723). They're the modern, recommended approach for custom TRL training.

## What are UV Scripts?

UV scripts declare dependencies at the top of the file using special comment syntax:

```python
# /// script
# dependencies = [
#     "trl>=0.12.0",
#     "transformers>=4.36.0",
# ]
# ///

# Your training code here
from trl import SFTTrainer
```

## Benefits

1. **Self-contained**: Dependencies are part of the script
2. **Version control**: Pin exact versions for reproducibility
3. **No setup files**: No requirements.txt or setup.py needed
4. **Portable**: Run anywhere UV is installed
5. **Clean**: Much cleaner than bash + pip + python strings

## Creating a UV Script

### Step 1: Define Dependencies

Start with dependency declaration:

```python
# /// script
# dependencies = [
#     "trl>=0.12.0",              # TRL for training
#     "transformers>=4.36.0",     # Transformers library
#     "datasets>=2.14.0",         # Dataset loading
#     "accelerate>=0.24.0",       # Distributed training
#     "peft>=0.7.0",              # LoRA/PEFT (optional)
# ]
# ///
```

### Step 2: Add Training Code

```python
# /// script
# dependencies = ["trl", "peft"]
# ///

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# Load dataset
dataset = load_dataset("trl-lib/Capybara", split="train")

# Configure training
config = SFTConfig(
    output_dir="my-model",
    num_train_epochs=3,
    push_to_hub=True,
    hub_model_id="username/my-model",
)

# Train
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,
    args=config,
    peft_config=LoraConfig(r=16, lora_alpha=32),
)

trainer.train()
trainer.push_to_hub()
```

### Step 3: Run on Jobs

```python
hf_jobs("uv", {
    "script": "train.py",  # or URL
    "flavor": "a10g-large",
    "timeout": "2h",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```

## Running Scripts from URLs

UV scripts can be run directly from URLs:

```python
hf_jobs("uv", {
    "script": "https://gist.github.com/username/abc123/raw/train.py",
    "flavor": "a10g-large",
    "timeout": "2h",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```

**Benefits:**
- Share scripts via GitHub Gists
- Version control in Git repos
- Scripts accessible from anywhere

## Working with Local Scripts

⚠️ **Important:** The `hf_jobs("uv", ...)` command does NOT support local file paths directly. You must make scripts accessible via URL.

### Why Local Paths Don't Work

The Jobs API runs in isolated Docker containers without access to your local filesystem. Scripts must be:
- Publicly accessible URLs, OR
- Accessible via authentication (HF_TOKEN for private repos)

**Don't:**
```python
# ❌ These will all fail
hf_jobs("uv", {"script": "train.py"})
hf_jobs("uv", {"script": "./scripts/train.py"})
hf_jobs("uv", {"script": "/path/to/train.py"})
```

**Do:**
```python
# ✅ These work
hf_jobs("uv", {"script": "https://huggingface.co/user/repo/resolve/main/train.py"})
hf_jobs("uv", {"script": "https://raw.githubusercontent.com/user/repo/main/train.py"})
hf_jobs("uv", {"script": "https://gist.githubusercontent.com/user/id/raw/train.py"})
```

### Recommended: Upload to Hugging Face Hub

The easiest way to use local scripts is to upload them to a Hugging Face repository:

```bash
# Create a dedicated scripts repo
huggingface-cli repo create my-training-scripts --type model

# Upload your script
huggingface-cli upload my-training-scripts ./train.py train.py

# If you update the script later
huggingface-cli upload my-training-scripts ./train.py train.py --commit-message "Updated training params"

# Use in jobs
script_url = "https://huggingface.co/USERNAME/my-training-scripts/resolve/main/train.py"

hf_jobs("uv", {
    "script": script_url,
    "flavor": "a10g-large",
    "timeout": "2h",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```

**Benefits:**
- ✅ Version control via Git
- ✅ Private repos supported (with HF_TOKEN)
- ✅ Easy to share and update
- ✅ No external dependencies
- ✅ Integrates with HF ecosystem

**For Private Scripts:**
```python
# Your script is in a private repo
hf_jobs("uv", {
    "script": "https://huggingface.co/USERNAME/private-scripts/resolve/main/train.py",
    "flavor": "a10g-large",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}  # Allows access to private repo
})
```

### Alternative: GitHub Gist

For quick scripts or one-off experiments:

```bash
# 1. Create a gist at https://gist.github.com
# 2. Paste your script
# 3. Click "Create public gist" (or secret gist)
# 4. Click the "Raw" button to get the raw URL

# Use in jobs
hf_jobs("uv", {
    "script": "https://gist.githubusercontent.com/username/gist-id/raw/train.py",
    "flavor": "a10g-large"
})
```

**Benefits:**
- ✅ Quick and easy
- ✅ No HF CLI setup needed
- ✅ Good for sharing examples

**Limitations:**
- ❌ Less version control than Git repos
- ❌ Secret gists are still publicly accessible via URL


## Using TRL Example Scripts

TRL provides maintained scripts that are UV-compatible:

```python
hf_jobs("uv", {
    "script": "https://raw.githubusercontent.com/huggingface/trl/main/examples/scripts/sft.py",
    "script_args": [
        "--model_name_or_path", "Qwen/Qwen2.5-0.5B",
        "--dataset_name", "trl-lib/Capybara",
        "--output_dir", "my-model",
        "--push_to_hub",
        "--hub_model_id", "username/my-model"
    ],
    "flavor": "a10g-large",
    "timeout": "2h",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```

**Available TRL scripts:**
- `sft.py` - Supervised fine-tuning
- `dpo.py` - Direct Preference Optimization
- `kto.py` - KTO training
- `grpo.py` - GRPO training
- `reward.py` - Reward model training
- `prm.py` - Process reward model

All at: https://github.com/huggingface/trl/tree/main/examples/scripts

## Best Practices

### 1. Pin Versions

Always pin dependency versions for reproducibility:

```python
# /// script
# dependencies = [
#     "trl==0.12.0",           # Exact version
#     "transformers>=4.36.0",  # Minimum version
# ]
# ///
```

### 2. Add Logging

Include progress logging for monitoring:

```python
print("✅ Dataset loaded")
print("🚀 Starting training...")
print(f"📊 Training on {len(dataset)} examples")
```

### 3. Validate Inputs

Check dataset and configuration before training:

```python
dataset = load_dataset("trl-lib/Capybara", split="train")
assert len(dataset) > 0, "Dataset is empty!"
print(f"✅ Dataset loaded: {len(dataset)} examples")
```

### 4. Add Comments

Document the script for future reference:

```python
# Train Qwen-0.5B on Capybara dataset using LoRA
# Expected runtime: ~2 hours on a10g-large
# Cost estimate: ~$6-8
```

### 5. Test Locally First

Test scripts locally before running on Jobs:

```bash
uv run train.py  # Runs locally with uv
```

## Docker Images

### Default Image

UV scripts run on default Python image with UV installed.

### TRL Image

Use official TRL image for faster startup:

```python
hf_jobs("uv", {
    "script": "train.py",
    "image": "huggingface/trl",  # Pre-installed TRL dependencies
    "flavor": "a10g-large",
    "timeout": "2h",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```

**Benefits:**
- Faster job startup (no pip install)
- All TRL dependencies pre-installed
- Tested and maintained by HF

## Template Scripts

### Basic SFT Template

```python
# /// script
# dependencies = ["trl>=0.12.0"]
# ///

from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

dataset = load_dataset("DATASET_NAME", split="train")

trainer = SFTTrainer(
    model="MODEL_NAME",
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="OUTPUT_DIR",
        num_train_epochs=3,
        push_to_hub=True,
        hub_model_id="USERNAME/MODEL_NAME",
    )
)

trainer.train()
trainer.push_to_hub()
```

### SFT with LoRA Template

```python
# /// script
# dependencies = ["trl>=0.12.0", "peft>=0.7.0"]
# ///

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

dataset = load_dataset("DATASET_NAME", split="train")

trainer = SFTTrainer(
    model="MODEL_NAME",
    train_dataset=dataset,
    peft_config=LoraConfig(r=16, lora_alpha=32),
    args=SFTConfig(
        output_dir="OUTPUT_DIR",
        num_train_epochs=3,
        push_to_hub=True,
        hub_model_id="USERNAME/MODEL_NAME",
    )
)

trainer.train()
trainer.push_to_hub()
```

### DPO Template

```python
# /// script
# dependencies = ["trl>=0.12.0"]
# ///

from datasets import load_dataset
from transformers import AutoTokenizer
from trl import DPOTrainer, DPOConfig

model_name = "MODEL_NAME"
dataset = load_dataset("DATASET_NAME", split="train")
tokenizer = AutoTokenizer.from_pretrained(model_name)

trainer = DPOTrainer(
    model=model_name,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=DPOConfig(
        output_dir="OUTPUT_DIR",
        num_train_epochs=3,
        push_to_hub=True,
        hub_model_id="USERNAME/MODEL_NAME",
    )
)

trainer.train()
trainer.push_to_hub()
```

## Troubleshooting

### Issue: Dependencies not installing
**Check:** Verify dependency names and versions are correct

### Issue: Script not found
**Check:** Verify URL is accessible and points to raw file

### Issue: Import errors
**Solution:** Add missing dependencies to `dependencies` list

### Issue: Slow startup
**Solution:** Use `image="huggingface/trl"` for pre-installed dependencies

---
name: trl
description: This skill should be used when users want to train or fine-tune language models using TRL (Transformer Reinforcement Learning) on Hugging Face Jobs infrastructure. Covers SFT, DPO, GRPO, KTO, reward modeling, and PPO training methods, plus GGUF conversion for local deployment. Includes guidance on the TRL Jobs package, UV scripts with PEP 723 format, dataset preparation and validation, hardware selection, cost estimation, Trackio monitoring, Hub authentication, and model persistence. Should be invoked for tasks involving cloud GPU training, GGUF conversion, or when users mention training on Hugging Face Jobs without local GPU setup.
license: Complete terms in LICENSE.txt
---

# TRL Training on Hugging Face Jobs

## Overview

Train language models using TRL (Transformer Reinforcement Learning) on fully managed Hugging Face infrastructure. No local GPU setup required—models train on cloud GPUs and results are automatically saved to the Hugging Face Hub.

**TRL provides multiple training methods:**
- **SFT** (Supervised Fine-Tuning) - Standard instruction tuning
- **DPO** (Direct Preference Optimization) - Alignment from preference data
- **GRPO** (Group Relative Policy Optimization) - Online RL training
- **KTO** (Kahneman-Tversky Optimization) - Preference tuning without paired data
- **Reward Modeling** - Train reward models for RLHF
- **PPO** (Proximal Policy Optimization) - Classic RLHF method

**For detailed TRL method documentation:**
```python
hf_doc_search("your query", product="trl")
hf_doc_fetch("https://huggingface.co/docs/trl/sft_trainer")  # SFT
hf_doc_fetch("https://huggingface.co/docs/trl/dpo_trainer")  # DPO
# etc.
```

**See also:** `references/training_methods.md` for method overviews and selection guidance

## When to Use This Skill

Use this skill when users want to:
- Fine-tune language models on cloud GPUs without local infrastructure
- Train with TRL methods (SFT, DPO, GRPO, KTO, etc.)
- Run training jobs on Hugging Face Jobs infrastructure
- Convert trained models to GGUF for local deployment (Ollama, LM Studio, llama.cpp)
- Ensure trained models are permanently saved to the Hub
- Use modern workflows with optimized defaults

## Key Directives

When assisting with training jobs:

1. **Submit jobs directly with inline scripts** - The `script` parameter accepts Python code directly. Do NOT save to local files unless the user explicitly requests it. Pass the script content as a string to `hf_jobs()`.

2. **Always include Trackio** - Every training script should include Trackio for real-time monitoring. Use example scripts in `scripts/` as templates.

3. **Provide job details after submission** - After submitting, provide job ID, monitoring URL, estimated time, and note that the user can request status checks later.

4. **Use example scripts as templates** - Reference `scripts/train_sft_example.py`, `scripts/train_dpo_example.py`, etc. as starting points.

## Prerequisites Checklist

Before starting any training job, verify:

### ✅ **Account & Authentication**
- Hugging Face Account with [Pro](https://hf.co/pro), [Team](https://hf.co/enterprise), or [Enterprise](https://hf.co/enterprise) plan (Jobs require paid plan)
- Authenticated login: Check with `mcp__huggingface__hf_whoami()`
- **HF_TOKEN for Hub Push** ⚠️ CRITICAL - Training environment is ephemeral, must push to Hub or ALL training results are lost
- Token must have write permissions and is automatically available as `$HF_TOKEN` in job secrets

### ✅ **Dataset Requirements**
- Dataset must exist on Hub or be loadable via `datasets.load_dataset()`
- Format must match training method (SFT: "messages"/text/prompt-completion; DPO: chosen/rejected; GRPO: prompt-only)
- Use `scripts/validate_dataset.py` to verify format or `hf_doc_fetch("https://huggingface.co/docs/trl/dataset_formats")` for complete reference
- Size appropriate for hardware (Demo: 50-100 examples on t4-small; Production: 1K-10K+ on a10g-large/a100-large)

### ⚠️ **Critical Settings**
- **Timeout must exceed expected training time** - Default 30min is TOO SHORT for most training. Minimum recommended: 1-2 hours. Job fails and loses all progress if timeout is exceeded.
- **Hub push must be enabled** - Config: `push_to_hub=True`, `hub_model_id="username/model-name"`; Job: `secrets={"HF_TOKEN": "$HF_TOKEN"}`

## Asynchronous Job Guidelines

**⚠️ IMPORTANT: Training jobs run asynchronously and can take hours**

### Action Required

**When user requests training:**
1. **Create the training script** with Trackio included (use `scripts/train_sft_example.py` as template)
2. **Submit immediately** using `hf_jobs()` MCP tool with script content inline - don't save to file unless user requests
3. **Report submission** with job ID, monitoring URL, and estimated time
4. **Wait for user** to request status checks - don't poll automatically

### Ground Rules
- **Jobs run in background** - Submission returns immediately; training continues independently
- **Initial logs delayed** - Can take 30-60 seconds for logs to appear
- **User checks status** - Wait for user to request status updates
- **Avoid polling** - Check logs only on user request; provide monitoring links instead

### After Submission

**Provide to user:**
- ✅ Job ID and monitoring URL
- ✅ Expected completion time
- ✅ Trackio dashboard URL
- ✅ Note that user can request status checks later

**Example Response:**
```
✅ Job submitted successfully!

Job ID: abc123xyz
Monitor: https://huggingface.co/jobs/username/abc123xyz

Expected time: ~2 hours
Estimated cost: ~$10

The job is running in the background. Ask me to check status/logs when ready!
```

## Quick Start: Three Approaches

### Approach 1: TRL Jobs Package (Easiest—Recommended for Beginners)

The `trl-jobs` package provides optimized defaults and one-liner training:

```bash
# Install (users only, not needed for this environment)
pip install trl-jobs

# Train with SFT (simplest possible)
trl-jobs sft \
  --model_name Qwen/Qwen2.5-0.5B \
  --dataset_name trl-lib/Capybara
```

**Benefits:** Pre-configured settings, automatic Trackio integration, automatic Hub push, one-line commands
**When to use:** User is new to training, standard scenarios, quick experimentation
**Repository:** https://github.com/huggingface/trl-jobs

### Approach 2: UV Scripts (Recommended for Custom Training)

UV scripts use PEP 723 inline dependencies for clean, self-contained training. **Submit script content directly inline:**

```python
hf_jobs("uv", {
    "script": """
# /// script
# dependencies = ["trl>=0.12.0", "peft>=0.7.0", "trackio"]
# ///

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import trackio

trackio.init(project="my-training", space_id="username/my-dashboard")

dataset = load_dataset("trl-lib/Capybara", split="train")

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,
    peft_config=LoraConfig(r=16, lora_alpha=32),
    args=SFTConfig(
        output_dir="my-model",
        push_to_hub=True,
        hub_model_id="username/my-model",
        num_train_epochs=3,
        report_to="trackio",
    )
)

trainer.train()
trainer.push_to_hub()
trackio.finish()
""",
    "flavor": "a10g-large",
    "timeout": "2h",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```

**Benefits:** Clean code, dependencies declared inline (PEP 723), no file saving required
**When to use:** Custom training logic, full control over training
**See:** `references/uv_scripts_guide.md` for complete UV scripts guide

### Approach 3: TRL Maintained Scripts (Run Official Examples)

TRL provides battle-tested scripts for all methods. Can be run from URLs:

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

**Benefits:** No code to write, maintained by TRL team, production-tested
**When to use:** Standard TRL training, quick experiments, don't need custom code
**Available:** sft.py, dpo.py, grpo.py, kto.py, reward.py, ppo.py - https://github.com/huggingface/trl/tree/main/examples/scripts

### Finding More UV Scripts on Hub

The `uv-scripts` organization provides ready-to-use UV scripts stored as datasets on Hugging Face Hub:

```python
# Discover available UV script collections
dataset_search({"author": "uv-scripts", "sort": "downloads", "limit": 20})

# Explore a specific collection
hub_repo_details(["uv-scripts/classification"], repo_type="dataset", include_readme=True)
```

**Popular collections:** ocr, classification, synthetic-data, vllm, dataset-creation

## Hardware Selection

| Model Size | Recommended Hardware | Cost (approx/hr) | Use Case |
|------------|---------------------|------------------|----------|
| <1B params | `t4-small` | ~$0.75 | Demos, quick tests |
| 1-3B params | `t4-medium`, `l4x1` | ~$1.50-2.50 | Development |
| 3-7B params | `a10g-small`, `a10g-large` | ~$3.50-5.00 | Production training |
| 7-13B params | `a10g-large`, `a100-large` | ~$5-10 | Large models (use LoRA) |
| 13B+ params | `a100-large`, `a10g-largex2` | ~$10-20 | Very large (use LoRA) |

**GPU Flavors:** cpu-basic/upgrade/performance/xl, t4-small/medium, l4x1/x4, a10g-small/large/largex2/largex4, a100-large, h100/h100x8

**Guidelines:**
- Use **LoRA/PEFT** for models >7B to reduce memory
- Multi-GPU automatically handled by TRL/Accelerate
- Start with smaller hardware for testing

**See:** `references/hardware_guide.md` for detailed specifications

## Critical: Saving Results to Hub

**⚠️ EPHEMERAL ENVIRONMENT—MUST PUSH TO HUB**

The Jobs environment is temporary. All files are deleted when the job ends. If the model isn't pushed to Hub, **ALL TRAINING IS LOST**.

### Required Configuration

**In training script/config:**
```python
SFTConfig(
    push_to_hub=True,
    hub_model_id="username/model-name",  # MUST specify
    hub_strategy="every_save",  # Optional: push checkpoints
)
```

**In job submission:**
```python
{
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}  # Enables authentication
}
```

### Verification Checklist

Before submitting:
- [ ] `push_to_hub=True` set in config
- [ ] `hub_model_id` includes username/repo-name
- [ ] `secrets` parameter includes HF_TOKEN
- [ ] User has write access to target repo

**See:** `references/hub_saving.md` for detailed troubleshooting

## Timeout Management

**⚠️ DEFAULT: 30 MINUTES—TOO SHORT FOR TRAINING**

### Setting Timeouts

```python
{
    "timeout": "2h"   # 2 hours (formats: "90m", "2h", "1.5h", or seconds as integer)
}
```

### Timeout Guidelines

| Scenario | Recommended | Notes |
|----------|-------------|-------|
| Quick demo (50-100 examples) | 10-30 min | Verify setup |
| Development training | 1-2 hours | Small datasets |
| Production (3-7B model) | 4-6 hours | Full datasets |
| Large model with LoRA | 3-6 hours | Depends on dataset |

**Always add 20-30% buffer** for model/dataset loading, checkpoint saving, Hub push operations, and network delays.

**On timeout:** Job killed immediately, all unsaved progress lost, must restart from beginning

## Cost Estimation

**Offer to estimate cost when planning jobs with known parameters.** Use `scripts/estimate_cost.py`:

```bash
python scripts/estimate_cost.py \
  --model meta-llama/Llama-2-7b-hf \
  --dataset trl-lib/Capybara \
  --hardware a10g-large \
  --dataset-size 16000 \
  --epochs 3
```

Output includes estimated time, cost, recommended timeout (with buffer), and optimization suggestions.

**When to offer:** User planning a job, asks about cost/time, choosing hardware, job will run >1 hour or cost >$5

## Example Training Scripts

**Production-ready templates with all best practices:**

- **`scripts/train_sft_example.py`** - Complete SFT training with Trackio, LoRA, checkpoints
- **`scripts/train_dpo_example.py`** - DPO training for preference learning
- **`scripts/train_grpo_example.py`** - GRPO training for online RL

These scripts demonstrate proper Hub saving, Trackio integration, checkpoint management, and optimized parameters. Pass their content inline to `hf_jobs()` or use as templates for custom scripts.

## Monitoring and Tracking

**Trackio** provides real-time metrics visualization. See `references/trackio_guide.md` for complete setup guide.

**Key points:**
- Add `"trackio"` to dependencies
- Initialize with `trackio.init(project="name", space_id="username/dashboard")`
- Configure trainer with `report_to="trackio"`
- Call `trackio.finish()` after training

**Alternative:** Use `report_to="tensorboard"` for simpler setup (logs saved with model to Hub)

### Check Job Status

```python
# List all jobs
hf_jobs("ps")

# Inspect specific job
hf_jobs("inspect", {"job_id": "your-job-id"})

# View logs
hf_jobs("logs", {"job_id": "your-job-id"})
```

**Remember:** Wait for user to request status checks. Avoid polling repeatedly.

## Converting Models to GGUF

After training, convert models to **GGUF format** for use with llama.cpp, Ollama, LM Studio, and other local inference tools.

**What is GGUF:**
- Optimized for CPU/GPU inference with llama.cpp
- Supports quantization (4-bit, 5-bit, 8-bit) to reduce model size
- Compatible with Ollama, LM Studio, Jan, GPT4All, llama.cpp
- Typically 2-8GB for 7B models (vs 14GB unquantized)

**When to convert:**
- Running models locally with Ollama or LM Studio
- Reducing model size with quantization
- Deploying to edge devices
- Sharing models for local-first use

**See:** `references/gguf_conversion.md` for complete conversion guide, including production-ready conversion script, quantization options, hardware requirements, usage examples, and troubleshooting.

**Quick conversion:**
```python
hf_jobs("uv", {
    "script": "<see references/gguf_conversion.md for complete script>",
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

## Common Training Patterns

See `references/training_patterns.md` for detailed examples including:
- Quick demo (5-10 minutes)
- Production with checkpoints
- Multi-GPU training
- DPO training (preference learning)
- GRPO training (online RL)

## Troubleshooting

**Common issues:**
- Job times out → Increase timeout, reduce epochs/dataset, use smaller model/LoRA
- Model not saved to Hub → Check push_to_hub=True, hub_model_id, secrets=HF_TOKEN
- Out of Memory (OOM) → Reduce batch size, increase gradient accumulation, enable LoRA, use larger GPU
- Dataset format error → Check format docs, validate dataset with `scripts/validate_dataset.py`
- Import/module errors → Add PEP 723 header with dependencies, verify format
- Authentication errors → Check `mcp__huggingface__hf_whoami()`, token permissions, secrets parameter

**See:** `references/troubleshooting.md` for complete troubleshooting guide

## Resources

### References (In This Skill)
- `references/training_methods.md` - Overview of SFT, DPO, GRPO, KTO, PPO, Reward Modeling
- `references/training_patterns.md` - Common training patterns and examples
- `references/gguf_conversion.md` - Complete GGUF conversion guide
- `references/trackio_guide.md` - Trackio monitoring setup
- `references/uv_scripts_guide.md` - Complete UV scripts guide
- `references/hardware_guide.md` - Hardware specs and selection
- `references/hub_saving.md` - Hub authentication troubleshooting
- `references/troubleshooting.md` - Common issues and solutions

### Scripts (In This Skill)
- `scripts/train_sft_example.py` - Production SFT template
- `scripts/train_dpo_example.py` - Production DPO template
- `scripts/train_grpo_example.py` - Production GRPO template
- `scripts/validate_dataset.py` - Validate dataset format before training
- `scripts/estimate_cost.py` - Estimate time and cost (offer when appropriate)
- `scripts/convert_to_gguf.py` - Complete GGUF conversion script

### External Links
- [TRL Documentation](https://huggingface.co/docs/trl)
- [TRL Jobs Training Guide](https://huggingface.co/docs/trl/en/jobs_training)
- [TRL Jobs Package](https://github.com/huggingface/trl-jobs)
- [HF Jobs Documentation](https://huggingface.co/docs/huggingface_hub/guides/jobs)
- [TRL Example Scripts](https://github.com/huggingface/trl/tree/main/examples/scripts)
- [UV Scripts Guide](https://docs.astral.sh/uv/guides/scripts/)
- [UV Scripts Organization](https://huggingface.co/uv-scripts)

## Key Takeaways

1. **Submit scripts inline** - The `script` parameter accepts Python code directly; no file saving required unless user requests
2. **Jobs are asynchronous** - Don't wait/poll; let user check when ready
3. **Always set timeout** - Default 30 min is insufficient; minimum 1-2 hours recommended
4. **Always enable Hub push** - Environment is ephemeral; without push, all results lost
5. **Include Trackio** - Use example scripts as templates for real-time monitoring
6. **Offer cost estimation** - When parameters are known, use `scripts/estimate_cost.py`
7. **Three approaches available:** TRL Jobs package (easiest), UV scripts (custom, modern), TRL maintained scripts (official examples)
8. **Use doc-fetch/doc-search** for latest TRL documentation
9. **Validate dataset format** before training with `scripts/validate_dataset.py`
10. **Choose appropriate hardware** for model size; use LoRA for models >7B

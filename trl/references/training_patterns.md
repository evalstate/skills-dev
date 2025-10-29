# Common Training Patterns

This guide provides common training patterns and use cases for TRL on Hugging Face Jobs.

## Quick Demo (5-10 minutes)

Test setup with minimal training:

```python
hf_jobs("uv", {
    "script": "https://raw.githubusercontent.com/huggingface/trl/main/examples/scripts/sft.py",
    "script_args": [
        "--model_name_or_path", "Qwen/Qwen2.5-0.5B",
        "--dataset_name", "trl-lib/Capybara",
        "--dataset_train_split", "train[:50]",  # Only 50 examples
        "--max_steps", "10",
        "--output_dir", "demo",
        "--push_to_hub",
        "--hub_model_id", "username/demo"
    ],
    "flavor": "t4-small",
    "timeout": "15m",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```

**Note:** The TRL maintained script above doesn't include Trackio. For production training with monitoring, see `scripts/train_sft_example.py` for a complete template with Trackio integration.

## Production with Checkpoints

Full training with intermediate saves. Use this pattern for long training runs where you want to save progress:

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

trackio.init(project="production-training", space_id="username/my-dashboard")

dataset = load_dataset("trl-lib/Capybara", split="train")

config = SFTConfig(
    output_dir="my-model",
    push_to_hub=True,
    hub_model_id="username/my-model",
    hub_strategy="every_save",  # Push each checkpoint
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    num_train_epochs=3,
    report_to="trackio",
)

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,
    args=config,
    peft_config=LoraConfig(r=16, lora_alpha=32),
)

trainer.train()
trainer.push_to_hub()
trackio.finish()
""",
    "flavor": "a10g-large",
    "timeout": "6h",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```

## Multi-GPU Training

Automatic distributed training across multiple GPUs. TRL/Accelerate handles distribution automatically:

```python
hf_jobs("uv", {
    "script": """
# Your training script here (same as single GPU)
# No changes needed - Accelerate detects multiple GPUs
""",
    "flavor": "a10g-largex2",  # 2x A10G GPUs
    "timeout": "4h",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```

**Tips for multi-GPU:**
- No code changes needed
- Use `per_device_train_batch_size` (per GPU, not total)
- Effective batch size = `per_device_train_batch_size` × `num_gpus` × `gradient_accumulation_steps`
- Monitor GPU utilization to ensure both GPUs are being used

## DPO Training (Preference Learning)

Train with preference data for alignment:

```python
hf_jobs("uv", {
    "script": """
# /// script
# dependencies = ["trl>=0.12.0", "trackio"]
# ///

from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
import trackio

trackio.init(project="dpo-training", space_id="username/my-dashboard")

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

config = DPOConfig(
    output_dir="dpo-model",
    push_to_hub=True,
    hub_model_id="username/dpo-model",
    num_train_epochs=1,
    beta=0.1,  # KL penalty coefficient
    report_to="trackio",
)

trainer = DPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",  # Use instruct model as base
    train_dataset=dataset,
    args=config,
)

trainer.train()
trainer.push_to_hub()
trackio.finish()
""",
    "flavor": "a10g-large",
    "timeout": "3h",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```

**For DPO documentation:** Use `hf_doc_fetch("https://huggingface.co/docs/trl/dpo_trainer")`

## GRPO Training (Online RL)

Group Relative Policy Optimization for online reinforcement learning:

```python
hf_jobs("uv", {
    "script": "https://raw.githubusercontent.com/huggingface/trl/main/examples/scripts/grpo.py",
    "script_args": [
        "--model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct",
        "--dataset_name", "trl-lib/math_shepherd",
        "--output_dir", "grpo-model",
        "--push_to_hub",
        "--hub_model_id", "username/grpo-model"
    ],
    "flavor": "a10g-large",
    "timeout": "4h",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```

**For GRPO documentation:** Use `hf_doc_fetch("https://huggingface.co/docs/trl/grpo_trainer")`

## Pattern Selection Guide

| Use Case | Pattern | Hardware | Time |
|----------|---------|----------|------|
| Test setup | Quick Demo | t4-small | 5-10 min |
| Small dataset (<1K) | Production w/ Checkpoints | t4-medium | 30-60 min |
| Medium dataset (1-10K) | Production w/ Checkpoints | a10g-large | 2-6 hours |
| Large dataset (>10K) | Multi-GPU | a10g-largex2 | 4-12 hours |
| Preference learning | DPO Training | a10g-large | 2-4 hours |
| Online RL | GRPO Training | a10g-large | 3-6 hours |

## Best Practices

1. **Always start with Quick Demo** - Verify setup before long runs
2. **Use checkpoints for runs >1 hour** - Protect against failures
3. **Enable Trackio** - Monitor progress in real-time
4. **Add 20-30% buffer to timeout** - Account for loading/saving overhead
5. **Test with small dataset slice first** - Use `"train[:100]"` to verify code
6. **Use multi-GPU for large models** - 7B+ models benefit significantly

## See Also

- `scripts/train_sft_example.py` - Complete SFT template with Trackio
- `scripts/train_dpo_example.py` - Complete DPO template
- `scripts/train_grpo_example.py` - Complete GRPO template
- `references/hardware_guide.md` - Detailed hardware specifications
- `references/training_methods.md` - Overview of all TRL training methods

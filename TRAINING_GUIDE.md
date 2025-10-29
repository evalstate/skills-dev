# Production Training Script Guide

## Quick Start

**Run with defaults (5-10 minute demo):**
```python
hf_jobs("uv", {
    "script": "https://huggingface.co/evalstate/demo-training-scripts/resolve/main/train_production_documented.py",
    "flavor": "t4-small",
    "timeout": "20m",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```

## Customizable Parameters

All parameters can be customized via environment variables without modifying the script:

### 🎯 Most Common Settings

| Parameter | Default | Description | When to Change |
|-----------|---------|-------------|----------------|
| `MODEL` | `Qwen/Qwen2.5-0.5B` | Model to fine-tune | Use larger model for better quality |
| `DATASET` | `trl-lib/Capybara` | Training dataset | Use your own dataset |
| `OUTPUT_REPO` | `evalstate/qwen-capybara-sft` | Where to save | Always set to your username |
| `MAX_STEPS` | `20` | Training duration | 100+ for real training |
| `LEARNING_RATE` | `2e-5` | Learning rate | Tune if loss unstable |

### 📊 Monitoring

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USE_TRACKIO` | `true` | Enable real-time monitoring | Set to `false` for faster demo |

### ⚙️ Advanced Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | `2` | Batch size per GPU | Increase for larger GPUs |
| `LEARNING_RATE` | `2e-5` | Learning rate | Typical range: 1e-5 to 5e-5 |

## Usage Examples

### Example 1: Quick Demo (Default)
```python
hf_jobs("uv", {
    "script": "train_production_documented.py",
    "flavor": "t4-small",
    "timeout": "20m",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```

**Result:** Trains Qwen-0.5B for 20 steps (~10 minutes, ~$0.20)

### Example 2: Custom Model & Dataset
```python
hf_jobs("uv", {
    "script": "train_production_documented.py",
    "flavor": "a10g-large",
    "timeout": "2h",
    "env": {
        "MODEL": "meta-llama/Llama-3.2-1B",
        "DATASET": "HuggingFaceH4/ultrachat_200k",
        "OUTPUT_REPO": "your-username/llama-ultrachat",
        "MAX_STEPS": "100"
    },
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```

**Result:** Trains Llama-3.2-1B for 100 steps (~1 hour, ~$5)

### Example 3: Longer Training Run
```python
hf_jobs("uv", {
    "script": "train_production_documented.py",
    "flavor": "a10g-large",
    "timeout": "6h",
    "env": {
        "MODEL": "Qwen/Qwen2.5-3B",
        "DATASET": "your-username/my-dataset",
        "OUTPUT_REPO": "your-username/qwen3b-custom",
        "MAX_STEPS": "500",
        "BATCH_SIZE": "4",
        "LEARNING_RATE": "1e-5"
    },
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```

**Result:** Production training (~4-5 hours, ~$20-25)

### Example 4: Without Monitoring (Fastest)
```python
hf_jobs("uv", {
    "script": "train_production_documented.py",
    "flavor": "t4-small",
    "timeout": "15m",
    "env": {
        "USE_TRACKIO": "false",
        "OUTPUT_REPO": "your-username/quick-test"
    },
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```

**Result:** Fastest possible demo (~8 minutes)

## How It Works

### Three Configuration Levels

```
┌─────────────────────────────────────┐
│ 1. CUSTOMIZABLE via env vars       │  ← Change these freely
│    MODEL, DATASET, MAX_STEPS, etc. │
├─────────────────────────────────────┤
│ 2. FIXED (edit script if needed)   │  ← Advanced users only
│    LORA_R, GRADIENT_ACCUMULATION    │
├─────────────────────────────────────┤
│ 3. TRAINING LOGIC                   │  ← Don't modify
│    Trainer initialization, etc.     │
└─────────────────────────────────────┘
```

### Single Trackio Space

All experiments go to **one dashboard**: `evalstate/ml-experiments`

- ✅ Easy comparison across experiments
- ✅ Filtered by project name
- ✅ No space clutter

**Access your dashboard:**
https://huggingface.co/spaces/evalstate/ml-experiments

## Recommended Models

| Model | Size | Speed | Use Case | Hardware |
|-------|------|-------|----------|----------|
| `Qwen/Qwen2.5-0.5B` | 0.5B | ⚡⚡⚡ | Quick tests | t4-small |
| `HuggingFaceTB/SmolLM2-1.7B` | 1.7B | ⚡⚡ | Development | t4-medium |
| `meta-llama/Llama-3.2-1B` | 1B | ⚡⚡ | Production | a10g-small |
| `Qwen/Qwen2.5-3B` | 3B | ⚡ | High quality | a10g-large |

## Recommended Datasets

**Conversational (chat/instruction):**
- `trl-lib/Capybara` - High-quality chat (16K examples)
- `HuggingFaceH4/ultrachat_200k` - Diverse conversations
- `argilla/distilabel-capybara-dpo-7k-binarized` - Preference data (for DPO)

**Your own dataset:**
- Must have `"messages"` field in conversational format
- See: https://huggingface.co/docs/trl/dataset_formats

## Hardware Selection

| Hardware | Cost/hr | When to Use |
|----------|---------|-------------|
| `t4-small` | ~$0.75 | Quick demos (20 steps) |
| `t4-medium` | ~$1.50 | Small models, testing |
| `a10g-small` | ~$3.50 | Production (1-3B models) |
| `a10g-large` | ~$5.00 | Production (3-7B models) |

## Timeout Guidelines

| Scenario | Recommended Timeout |
|----------|-------------------|
| Quick demo (20 steps) | 15-20m |
| Development (100 steps) | 1-2h |
| Production (500+ steps) | 4-6h |

**Always add 20-30% buffer** for setup time!

## Troubleshooting

### "Out of memory"
**Solution:** Reduce `BATCH_SIZE` or use larger GPU
```python
"env": {"BATCH_SIZE": "1"}  # Smallest possible
```

### "Dataset not found"
**Check:** Dataset name is correct and public
```python
"env": {"DATASET": "username/dataset-name"}
```

### "Permission denied pushing to Hub"
**Check:** `OUTPUT_REPO` uses your username
```python
"env": {"OUTPUT_REPO": "YOUR-USERNAME/model-name"}
```

### Training too slow/expensive
**Solution:** Reduce steps or use smaller model
```python
"env": {
    "MAX_STEPS": "10",  # Faster
    "MODEL": "Qwen/Qwen2.5-0.5B"  # Smaller
}
```

## Next Steps

After training completes:

1. **View your model:** https://huggingface.co/YOUR-USERNAME/model-name
2. **Check metrics:** https://huggingface.co/spaces/evalstate/ml-experiments
3. **Test the model:** Use the Inference API or download locally
4. **Share:** Model is public by default (or make private in settings)

## Questions?

- 📖 Full TRL docs: https://huggingface.co/docs/trl
- 💬 Ask in Hugging Face Discord
- 🐛 Issues: Check job logs for error messages

# Training Script Comparison

## What Makes the Production Script "Clean"?

### ❌ Original Script Issues

```python
# 1. Double Trackio initialization
trackio.init(project="qwen-demo-sft", space_id="evalstate/trackio-demo")
# ... later ...
SFTConfig(report_to="trackio")  # TRL initializes AGAIN!
# Result: 2 spaces created!

# 2. Unclear what's customizable
model="Qwen/Qwen2.5-0.5B"  # Hard to find/change
dataset = load_dataset("trl-lib/Capybara")  # Buried in code

# 3. Mixed concerns
trackio.init(...)
dataset = load_dataset(...)
print(f"Sample: {dataset[0]}")  # Debugging mixed with setup
config = SFTConfig(...)
```

### ✅ Clean Production Script

```python
# 1. Single Trackio init
trackio.init(...)
SFTConfig(report_to="trackio")  # Uses existing connection
# Result: 1 space!

# 2. Clear configuration section at top
# ============================================================================
# CONFIGURATION - Customize via environment variables
# ============================================================================
MODEL = os.getenv("MODEL", "Qwen/Qwen2.5-0.5B")
DATASET = os.getenv("DATASET", "trl-lib/Capybara")
# All customizable params grouped together with comments

# 3. Separated concerns
# Setup
if USE_TRACKIO:
    trackio.init(...)

# Load data
dataset = load_dataset(...)

# Train
trainer = SFTTrainer(...)
trainer.train()
```

## Key Improvements

### 1. Documentation Structure

```python
"""
CUSTOMIZABLE PARAMETERS (via environment variables):
    MODEL           - Model to fine-tune (default: Qwen/Qwen2.5-0.5B)
    DATASET         - Dataset name on Hub (default: trl-lib/Capybara)
    ...

EXAMPLE USAGE:
    hf_jobs("uv", {
        "env": {"MODEL": "meta-llama/Llama-3.2-1B"}
    })
"""
```

**Benefits:**
- ✅ Clear what can be changed
- ✅ Shows how to change it
- ✅ Includes examples
- ✅ Visible at file top

### 2. Three-Tier Configuration

```
┌─────────────────────────────────────┐
│ TIER 1: Environment Variables       │
│ ✅ Change freely without editing    │
│ MODEL, DATASET, MAX_STEPS, etc.     │
├─────────────────────────────────────┤
│ TIER 2: Fixed Constants             │
│ ⚙️  Edit if needed (advanced)       │
│ LORA_R, GRADIENT_ACCUMULATION       │
├─────────────────────────────────────┤
│ TIER 3: Training Logic              │
│ 🔒 Don't modify                     │
│ Trainer initialization, etc.        │
└─────────────────────────────────────┘
```

**Benefits:**
- ✅ Clear what to change vs what to leave alone
- ✅ Beginners change env vars only
- ✅ Advanced users can modify Tier 2
- ✅ Tier 3 is implementation detail

### 3. Inline Documentation

```python
# Model Selection
MODEL = os.getenv("MODEL", "Qwen/Qwen2.5-0.5B")
# Common options:
#   - Qwen/Qwen2.5-0.5B (fast, demo)
#   - Qwen/Qwen2.5-3B (production)
#   - meta-llama/Llama-3.2-1B
#   - HuggingFaceTB/SmolLM2-1.7B
```

**Benefits:**
- ✅ Context right where you need it
- ✅ Shows concrete examples
- ✅ Explains when to use each option

### 4. Clear Output

```python
print("="*80)
print("🚀 TRAINING CONFIGURATION")
print("="*80)
print(f"Model:          {MODEL}")
print(f"Dataset:        {DATASET}")
...
```

**Benefits:**
- ✅ Easy to verify settings before training
- ✅ Appears in job logs
- ✅ Helpful for debugging
- ✅ Professional appearance

### 5. Single Monitoring Space

```python
# All projects use same space
TRACKIO_SPACE = "evalstate/ml-experiments"

# Different project names for filtering
project=OUTPUT_REPO.split('/')[-1]  # e.g., "qwen-demo"
```

**Benefits:**
- ✅ One dashboard for all experiments
- ✅ Easy comparison across projects
- ✅ Filter by project name
- ✅ No space proliferation

## Usage Comparison

### Original (Hard to Customize)

```python
# To change model, must edit script:
# 1. Download script
# 2. Edit: model="new-model"
# 3. Upload to Hub
# 4. Submit job

hf_jobs("uv", {
    "script": "train.py",  # Your modified version
    "flavor": "t4-small"
})
```

### Production (Easy to Customize)

```python
# Just pass env vars:
hf_jobs("uv", {
    "script": "train_production_documented.py",  # Same script!
    "flavor": "a10g-large",
    "env": {
        "MODEL": "meta-llama/Llama-3.2-1B",
        "MAX_STEPS": "100"
    }
})
```

**No script editing needed!**

## File Summary

We created three files:

1. **`train_production_documented.py`** - The main script
   - 150 lines (well-commented)
   - Clear three-tier structure
   - Inline documentation
   - Ready to use

2. **`TRAINING_GUIDE.md`** - Usage guide
   - Quick start examples
   - Parameter reference
   - Troubleshooting
   - Best practices

3. **`COMPARISON.md`** (this file) - Design rationale
   - Why changes were made
   - Before/after comparison
   - Benefits explained

## When to Use Each Approach

| Approach | When to Use |
|----------|-------------|
| **Minimal** | Learning, one-off tests |
| **Production** | Reusable experiments, multiple runs |
| **Organized** | Team projects, complex workflows |

## Next Steps

1. **Upload to Hub:**
   ```bash
   hf upload evalstate/demo-training-scripts train_production_documented.py
   hf upload evalstate/demo-training-scripts TRAINING_GUIDE.md
   ```

2. **Run it:**
   ```python
   hf_jobs("uv", {
       "script": "https://huggingface.co/evalstate/demo-training-scripts/resolve/main/train_production_documented.py",
       "flavor": "t4-small",
       "timeout": "20m",
       "secrets": {"HF_TOKEN": "$HF_TOKEN"}
   })
   ```

3. **Customize it:**
   ```python
   "env": {"MODEL": "your-model", "MAX_STEPS": "50"}
   ```

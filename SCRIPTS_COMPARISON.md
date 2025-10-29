# Scripts Comparison: Current Directory vs TRL Skill

## Two Different Locations

### 1. Current Directory Scripts (What we just created)
**Location:** `/home/ssmith/source/training/.claude/skills/`

These are the training scripts we just created during this session:

| Script | Lines | Purpose |
|--------|-------|---------|
| `train_minimal.py` | 25 | Learning/demo - bare minimum |
| `train_production.py` | 75 | Reusable - env vars, no docs |
| `train_production_documented.py` | 150 | Reusable - env vars + extensive docs ⭐ |
| `train_organized.py` | 140 | Enterprise - functions, class-based |
| `train_demo.py` | 80 | Original demo (has Trackio bug) |

### 2. TRL Skill Scripts (Pre-existing)
**Location:** `/home/ssmith/source/training/.claude/skills/trl/scripts/`

These are part of the TRL skill and were already there:

| Script | Lines | Purpose |
|--------|-------|---------|
| `train_sft_example.py` | 111 | SFT training example (from skill) |
| `train_dpo_example.py` | 98 | DPO training example (from skill) |
| `train_grpo_example.py` | 97 | GRPO training example (from skill) |
| `validate_dataset.py` | 175 | Dataset validation utility |
| `estimate_cost.py` | 149 | Cost estimation utility |
| `convert_to_gguf.py` | 301 | GGUF conversion utility |

## Key Differences

### Purpose

**Current Directory (our new scripts):**
- Created during THIS conversation
- Demonstrate different levels of documentation
- Show progression from simple → documented → organized
- Teaching examples for "clean code"

**TRL Skill Scripts:**
- Part of the pre-existing TRL skill
- Production examples for different training methods
- Utility scripts for common tasks
- Reference implementations

### Focus

**Current Directory:**
```
train_minimal.py           → "How simple can we go?"
train_production.py        → "Add configurability"
train_production_documented.py → "Add documentation" ⭐
train_organized.py         → "Add structure"
```
*Shows evolution of code quality*

**TRL Skill Scripts:**
```
train_sft_example.py    → "How to do SFT"
train_dpo_example.py    → "How to do DPO"
train_grpo_example.py   → "How to do GRPO"
validate_dataset.py     → "Utility: validate format"
estimate_cost.py        → "Utility: estimate cost"
```
*Shows different training methods + utilities*

## Detailed Comparison: Our Scripts vs TRL Skill's train_sft_example.py

### trl/scripts/train_sft_example.py (111 lines)
```python
# From the TRL skill - complete production example
import trackio
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# Initialize Trackio
trackio.init(
    project="qwen-capybara-sft",
    space_id="username/my-trackio-dashboard",
    config={...}
)

# Load dataset
dataset = load_dataset("trl-lib/Capybara", split="train")

# Configure training
config = SFTConfig(
    output_dir="qwen-capybara-sft",
    push_to_hub=True,
    hub_model_id="username/qwen-capybara-sft",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    # ... many parameters explicitly set
)

# Train
trainer = SFTTrainer(...)
trainer.train()
trainer.push_to_hub()
```

**Characteristics:**
- ✅ Complete production example
- ✅ Includes Trackio (single init)
- ✅ Full dataset (not limited to 50)
- ✅ Many parameters explicitly set
- ❌ NOT configurable via env vars
- ❌ Hard-coded values (must edit to change)
- ❌ Minimal inline documentation

### Our train_production_documented.py (150 lines)
```python
# Our new script - configurable + documented
"""
CUSTOMIZABLE PARAMETERS (via environment variables):
    MODEL           - Model to fine-tune (default: Qwen/Qwen2.5-0.5B)
    DATASET         - Dataset name on Hub (default: trl-lib/Capybara)
    ...
"""

import os
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# CONFIGURATION - Customize via environment variables
MODEL = os.getenv("MODEL", "Qwen/Qwen2.5-0.5B")
# Common options:
#   - Qwen/Qwen2.5-0.5B (fast, demo)
#   - Qwen/Qwen2.5-3B (production)

DATASET = os.getenv("DATASET", "trl-lib/Capybara")
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
USE_TRACKIO = os.getenv("USE_TRACKIO", "true").lower() == "true"

# ... rest of training code
```

**Characteristics:**
- ✅ Environment-based configuration
- ✅ Extensive inline documentation
- ✅ Examples in comments
- ✅ Reusable without editing
- ✅ Clear sections
- ✅ Optional Trackio
- ⚠️  Smaller dataset (50 examples) for demo

## When to Use What?

### Use TRL Skill Scripts When:
- ✅ You want complete production examples
- ✅ You need reference implementations
- ✅ You want to see different training methods (SFT, DPO, GRPO)
- ✅ You need utility scripts (validation, cost estimation)
- ✅ You're okay editing the script to customize

### Use Our New Scripts When:
- ✅ You want quick demos (train_minimal.py)
- ✅ You need reusable scripts (train_production.py)
- ✅ You want self-documenting code (train_production_documented.py)
- ✅ You need to run many experiments with different settings
- ✅ You want to customize via environment variables
- ✅ You're sharing with team and want clear documentation

## Relationship

```
TRL Skill Scripts (Reference)
     ↓
     └─ train_sft_example.py (111 lines, production example)
     
Our New Scripts (Teaching Progression)
     ↓
     ├─ train_minimal.py (25 lines, bare minimum)
     ├─ train_production.py (75 lines, configurable)
     ├─ train_production_documented.py (150 lines, configurable + docs) ⭐
     └─ train_organized.py (140 lines, structured)
```

## Summary

**TRL Skill Scripts:**
- Part of the TRL skill infrastructure
- Reference implementations
- Show "what to do"
- Must edit to customize

**Our New Scripts:**
- Created during this conversation
- Teaching progression (simple → documented → organized)
- Show "how to organize code"
- Customize via environment variables

**They complement each other!**
- TRL scripts show the different training methods
- Our scripts show different code organization styles

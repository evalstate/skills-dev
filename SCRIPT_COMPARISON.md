# Training Scripts Comparison

## All Scripts in Directory

| Script | Lines | Purpose | Best For |
|--------|-------|---------|----------|
| **train_minimal.py** | ~25 | Absolute bare minimum | Learning, throw-away tests |
| **train_production.py** | ~75 | Env-configurable, clean | Reusable experiments |
| **train_production_documented.py** | ~150 | Same as production + docs | Sharing with others |
| **train_organized.py** | ~140 | Functions, class-based | Complex team projects |
| **train_demo.py** | ~80 | Original demo script | Initial demo (has issues) |

## Quick Comparison

### 1. train_minimal.py (25 lines)
```python
# Simplest possible
from trl import SFTTrainer, SFTConfig

dataset = load_dataset("trl-lib/Capybara", split="train[:50]")

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="output",
        push_to_hub=True,
        hub_model_id="evalstate/qwen-demo-minimal",
        max_steps=20,
        report_to="none",  # No monitoring
    )
)
trainer.train()
```

**Pros:**
- ✅ Super simple
- ✅ Easy to understand
- ✅ Fast execution (no monitoring overhead)

**Cons:**
- ❌ Hard to customize (must edit code)
- ❌ No monitoring
- ❌ No environment variables

**Use when:**
- Learning TRL basics
- One-time quick test
- Don't care about metrics

---

### 2. train_production.py (75 lines)
```python
# Environment-configurable
MODEL = os.getenv("MODEL", "Qwen/Qwen2.5-0.5B")
DATASET = os.getenv("DATASET", "trl-lib/Capybara")
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
USE_TRACKIO = os.getenv("USE_TRACKIO", "true").lower() == "true"

if USE_TRACKIO:
    trackio.init(
        project=OUTPUT_REPO.split('/')[-1],
        space_id="evalstate/ml-experiments",
        config={...}
    )

trainer = SFTTrainer(...)
trainer.train()
```

**Pros:**
- ✅ Reusable (change via env vars)
- ✅ Trackio monitoring
- ✅ Clean structure
- ✅ Production-ready

**Cons:**
- ❌ Less documentation in code
- ❌ Need to know which env vars exist

**Use when:**
- Running multiple experiments
- Need to compare different models/datasets
- Want monitoring

---

### 3. train_production_documented.py (150 lines) ⭐ RECOMMENDED
```python
"""
CUSTOMIZABLE PARAMETERS (via environment variables):
    MODEL           - Model to fine-tune (default: Qwen/Qwen2.5-0.5B)
    DATASET         - Dataset name on Hub (default: trl-lib/Capybara)
    OUTPUT_REPO     - Where to save model (default: evalstate/qwen-capybara-sft)
    MAX_STEPS       - Training steps (default: 20)
    BATCH_SIZE      - Batch size per device (default: 2)
    LEARNING_RATE   - Learning rate (default: 2e-5)
    USE_TRACKIO     - Enable monitoring (default: true)
"""

# Model Selection
MODEL = os.getenv("MODEL", "Qwen/Qwen2.5-0.5B")
# Common options:
#   - Qwen/Qwen2.5-0.5B (fast, demo)
#   - Qwen/Qwen2.5-3B (production)
#   - meta-llama/Llama-3.2-1B

# Dataset Selection
DATASET = os.getenv("DATASET", "trl-lib/Capybara")
# Use any conversational dataset with "messages" field

# Training Parameters
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
# Quick demo: 10-20 | Development: 100-500 | Production: 1000+

# ... rest of training code (same as train_production.py)
```

**Pros:**
- ✅ Same functionality as train_production.py
- ✅ Self-documenting
- ✅ Shows examples in comments
- ✅ Easy for new users
- ✅ Professional looking

**Cons:**
- ❌ More lines (but mostly comments)

**Use when:**
- Sharing with team
- Need documentation
- Want examples inline
- **⭐ Default choice for most cases**

---

### 4. train_organized.py (140 lines)
```python
class Config:
    """Training configuration with environment overrides"""
    MODEL = os.getenv("MODEL", "Qwen/Qwen2.5-0.5B")
    DATASET = os.getenv("DATASET", "trl-lib/Capybara")
    # ... all config in one place

def setup_monitoring(config: Config):
    """Initialize Trackio for experiment tracking"""
    ...

def load_and_validate_dataset(config: Config):
    """Load dataset and perform basic validation"""
    ...

def main():
    config = Config()
    setup_monitoring(config)
    dataset = load_and_validate_dataset(config)
    trainer = train(dataset)
    ...
```

**Pros:**
- ✅ Cleanest separation of concerns
- ✅ Easy to test individual functions
- ✅ Best for complex workflows
- ✅ Team-friendly structure

**Cons:**
- ❌ More complex
- ❌ Overkill for simple training

**Use when:**
- Team project
- Need to extend/customize heavily
- Want to unit test components
- Complex training pipeline

---

### 5. train_demo.py (80 lines) - Original Demo
```python
# Has the double Trackio issue!
trackio.init(
    project="qwen-demo-sft",
    space_id="evalstate/trackio-demo",  # Creates space 1
    ...
)

config = SFTConfig(
    report_to="trackio",  # TRL creates space 2!
    ...
)
```

**Issues:**
- ❌ Creates 2 Trackio spaces
- ❌ Not configurable
- ❌ Mixed concerns

**This was our original script that we improved!**

---

## Decision Matrix

### Choose Based On Your Needs:

```
Need simplicity above all?
  → train_minimal.py

Running ONE experiment?
  → train_minimal.py or train_production.py

Running MULTIPLE experiments?
  → train_production_documented.py ⭐

Sharing with others?
  → train_production_documented.py ⭐

Complex team project?
  → train_organized.py

Just learning?
  → train_minimal.py
```

## Key Differences

### Customization Method

| Script | How to Customize |
|--------|------------------|
| train_minimal.py | Edit code |
| train_production.py | Environment variables |
| train_production_documented.py | Environment variables (with docs) |
| train_organized.py | Environment variables or Config class |

### Monitoring

| Script | Monitoring |
|--------|-----------|
| train_minimal.py | None |
| train_production.py | Trackio (optional) |
| train_production_documented.py | Trackio (optional) |
| train_organized.py | Trackio (optional) |
| train_demo.py | Trackio (with bug!) |

### Lines of Code

| Script | Lines | Actual Code | Comments/Docs |
|--------|-------|-------------|---------------|
| train_minimal.py | 25 | 20 | 5 |
| train_production.py | 75 | 60 | 15 |
| train_production_documented.py | 150 | 60 | 90 |
| train_organized.py | 140 | 100 | 40 |

**Note:** train_production.py and train_production_documented.py have the SAME actual code, just different amounts of documentation!

## The Winner: train_production_documented.py ⭐

### Why It's the Best Choice:

1. **Self-documenting** - Everything explained inline
2. **Easy to customize** - Just pass env vars
3. **Shows examples** - Comments show real options
4. **Professional** - Clean output formatting
5. **Single Trackio space** - No duplication
6. **Reusable** - Same script for all experiments

### What Makes It Different from train_production.py?

**Same code, more documentation!**

```python
# train_production.py (minimal comments)
MODEL = os.getenv("MODEL", "Qwen/Qwen2.5-0.5B")

# train_production_documented.py (helpful comments)
MODEL = os.getenv("MODEL", "Qwen/Qwen2.5-0.5B")
# Common options:
#   - Qwen/Qwen2.5-0.5B (fast, demo)
#   - Qwen/Qwen2.5-3B (production)
#   - meta-llama/Llama-3.2-1B
```

The actual execution is identical, but the documented version is much easier to understand and use!

## Usage Examples

### Minimal
```python
# Must edit script to change model
hf_jobs("uv", {"script": "train_minimal.py", "flavor": "t4-small"})
```

### Production (Either Version)
```python
# Change model via environment
hf_jobs("uv", {
    "script": "train_production_documented.py",
    "flavor": "a10g-large",
    "env": {
        "MODEL": "meta-llama/Llama-3.2-1B",
        "MAX_STEPS": "100"
    }
})
```

### Organized
```python
# Same as production, just different internal structure
hf_jobs("uv", {
    "script": "train_organized.py",
    "flavor": "a10g-large",
    "env": {"MODEL": "meta-llama/Llama-3.2-1B"}
})
```

## Recommendation

**Use `train_production_documented.py` for:**
- ✅ All production work
- ✅ Sharing with teammates
- ✅ Multiple experiments
- ✅ When you want documentation

**Use `train_minimal.py` for:**
- ✅ Quick learning
- ✅ Throwaway tests
- ✅ When you want absolute simplicity

**Use `train_organized.py` for:**
- ✅ Complex team projects
- ✅ When you need to heavily extend the code
- ✅ Unit testing individual components

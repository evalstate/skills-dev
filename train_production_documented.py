# /// script
# dependencies = ["trl>=0.12.0", "peft>=0.7.0", "trackio"]
# ///

"""
Production Training Script with Environment Configuration

CUSTOMIZABLE PARAMETERS (via environment variables):
    MODEL           - Model to fine-tune (default: Qwen/Qwen2.5-0.5B)
    DATASET         - Dataset name on Hub (default: trl-lib/Capybara)
    OUTPUT_REPO     - Where to save model (default: evalstate/qwen-capybara-sft)
    MAX_STEPS       - Training steps (default: 20)
    BATCH_SIZE      - Batch size per device (default: 2)
    LEARNING_RATE   - Learning rate (default: 2e-5)
    USE_TRACKIO     - Enable monitoring (default: true)

EXAMPLE USAGE:
    # Default (quick demo):
    hf_jobs("uv", {"script": "train_production.py", "flavor": "t4-small"})
    
    # Custom settings:
    hf_jobs("uv", {
        "script": "train_production.py",
        "flavor": "a10g-large",
        "env": {
            "MODEL": "meta-llama/Llama-3.2-1B",
            "MAX_STEPS": "100",
            "LEARNING_RATE": "1e-5"
        }
    })
"""

import os
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ============================================================================
# CONFIGURATION - Customize via environment variables
# ============================================================================

# Model Selection
MODEL = os.getenv("MODEL", "Qwen/Qwen2.5-0.5B")
# Common options:
#   - Qwen/Qwen2.5-0.5B (fast, demo)
#   - Qwen/Qwen2.5-3B (production)
#   - meta-llama/Llama-3.2-1B
#   - HuggingFaceTB/SmolLM2-1.7B

# Dataset Selection
DATASET = os.getenv("DATASET", "trl-lib/Capybara")
# Use any conversational dataset with "messages" field
# Examples: trl-lib/Capybara, HuggingFaceH4/ultrachat_200k

# Output Configuration
OUTPUT_REPO = os.getenv("OUTPUT_REPO", "evalstate/qwen-capybara-sft")
# Must be in format: username/model-name

# Training Parameters
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
# Quick demo: 10-20 | Development: 100-500 | Production: 1000+

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
# Adjust based on GPU memory: t4-small=2, a10g-large=4-8

LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))
# Typical range: 1e-5 to 5e-5

# Monitoring
USE_TRACKIO = os.getenv("USE_TRACKIO", "true").lower() == "true"
# Set to "false" to disable real-time monitoring

# ============================================================================
# FIXED CONFIGURATION - Advanced users can modify these directly
# ============================================================================

# LoRA Configuration (reduces memory usage)
LORA_R = 8              # Rank (higher = more parameters, better quality)
LORA_ALPHA = 16         # Scaling factor (typically 2x LORA_R)
LORA_DROPOUT = 0.05     # Dropout rate

# Training Advanced
GRADIENT_ACCUMULATION = 2   # Effective batch size = BATCH_SIZE * this
WARMUP_RATIO = 0.1          # Percentage of steps for warmup
LR_SCHEDULER = "cosine"     # Learning rate schedule

# Logging
LOGGING_STEPS = None    # Auto-calculated (MAX_STEPS // 4)

# Trackio Space (single dashboard for all experiments)
TRACKIO_SPACE = "evalstate/ml-experiments"

# ============================================================================
# TRAINING SCRIPT - No need to modify below this line
# ============================================================================

print("="*80)
print("🚀 TRAINING CONFIGURATION")
print("="*80)
print(f"Model:          {MODEL}")
print(f"Dataset:        {DATASET}")
print(f"Output:         {OUTPUT_REPO}")
print(f"Max Steps:      {MAX_STEPS}")
print(f"Batch Size:     {BATCH_SIZE}")
print(f"Learning Rate:  {LEARNING_RATE}")
print(f"Monitoring:     {'Trackio' if USE_TRACKIO else 'Disabled'}")
print(f"LoRA:           r={LORA_R}, alpha={LORA_ALPHA}")
print("="*80)
print()

# Setup monitoring if enabled
if USE_TRACKIO:
    import trackio
    trackio.init(
        project=OUTPUT_REPO.split('/')[-1],  # Use model name as project
        space_id=TRACKIO_SPACE,
        config={
            "model": MODEL,
            "dataset": DATASET,
            "max_steps": MAX_STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "lora_r": LORA_R,
        }
    )
    print(f"📊 Trackio Dashboard: https://huggingface.co/spaces/{TRACKIO_SPACE}")
    print()

# Load dataset (first 50 examples for demo)
print("📦 Loading dataset...")
dataset = load_dataset(DATASET, split="train[:50]")
print(f"✅ Loaded {len(dataset)} examples")
print()

# Configure training
logging_steps = LOGGING_STEPS or max(1, MAX_STEPS // 4)

config = SFTConfig(
    # Output
    output_dir="output",
    push_to_hub=True,
    hub_model_id=OUTPUT_REPO,
    hub_strategy="end",
    
    # Training params (customizable via env vars)
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    
    # Optimization
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type=LR_SCHEDULER,
    
    # Logging
    logging_steps=logging_steps,
    report_to="trackio" if USE_TRACKIO else "none",
    
    # No checkpoints for demo (saves time)
    save_strategy="no",
)

# LoRA configuration (reduces memory usage)
peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

# Initialize trainer
print("🔥 Initializing trainer...")
trainer = SFTTrainer(
    model=MODEL,
    train_dataset=dataset,
    args=config,
    peft_config=peft_config,
)

# Train
print("🏃 Training started...")
print()
trainer.train()
print()

# Save to Hub
print("💾 Pushing model to Hub...")
trainer.push_to_hub()

# Finish monitoring
if USE_TRACKIO:
    trackio.finish()

# Summary
print()
print("="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print(f"📦 Model:   https://huggingface.co/{OUTPUT_REPO}")
if USE_TRACKIO:
    print(f"📊 Metrics: https://huggingface.co/spaces/{TRACKIO_SPACE}")
print("="*80)

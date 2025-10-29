# /// script
# dependencies = ["trl>=0.12.0", "peft>=0.7.0", "trackio"]
# ///

import os
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# Configuration from environment (with sensible defaults)
MODEL = os.getenv("MODEL", "Qwen/Qwen2.5-0.5B")
DATASET = os.getenv("DATASET", "trl-lib/Capybara")
OUTPUT_REPO = os.getenv("OUTPUT_REPO", "evalstate/qwen-capybara-sft")
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))
USE_TRACKIO = os.getenv("USE_TRACKIO", "true").lower() == "true"

print(f"🚀 Training Configuration:")
print(f"   Model: {MODEL}")
print(f"   Dataset: {DATASET}")
print(f"   Output: {OUTPUT_REPO}")
print(f"   Max Steps: {MAX_STEPS}")
print(f"   Monitoring: {'Trackio' if USE_TRACKIO else 'None'}")

# Setup monitoring if enabled
if USE_TRACKIO:
    import trackio
    trackio.init(
        project=OUTPUT_REPO.split('/')[-1],  # Use model name as project
        space_id="evalstate/ml-experiments",  # Single space for all
        config={
            "model": MODEL,
            "dataset": DATASET,
            "max_steps": MAX_STEPS,
            "learning_rate": LEARNING_RATE,
        }
    )

# Load dataset
dataset = load_dataset(DATASET, split="train[:50]")
print(f"✅ Loaded {len(dataset)} examples")

# Configure training
config = SFTConfig(
    output_dir="output",
    push_to_hub=True,
    hub_model_id=OUTPUT_REPO,
    hub_strategy="end",
    
    # Training params
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,
    learning_rate=LEARNING_RATE,
    
    # Optimization
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    
    # Logging
    logging_steps=max(1, MAX_STEPS // 4),
    report_to="trackio" if USE_TRACKIO else "none",
    
    # No checkpoints for demo
    save_strategy="no",
)

# LoRA config
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Train
print("🔥 Starting training...")
trainer = SFTTrainer(
    model=MODEL,
    train_dataset=dataset,
    args=config,
    peft_config=peft_config,
)

trainer.train()
trainer.push_to_hub()

if USE_TRACKIO:
    trackio.finish()

print(f"✅ Complete! Model: https://huggingface.co/{OUTPUT_REPO}")
if USE_TRACKIO:
    print(f"📊 Metrics: https://huggingface.co/spaces/evalstate/ml-experiments")

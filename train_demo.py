# /// script
# dependencies = [
#     "trl>=0.12.0",
#     "peft>=0.7.0",
#     "transformers>=4.36.0",
#     "accelerate>=0.24.0",
#     "trackio",
# ]
# ///

import trackio
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# Initialize Trackio for real-time monitoring
trackio.init(
    project="qwen-demo-sft",
    space_id="evalstate/trackio-demo",  # Will auto-create if doesn't exist
    config={
        "model": "Qwen/Qwen2.5-0.5B",
        "dataset": "trl-lib/Capybara",
        "dataset_size": 50,
        "learning_rate": 2e-5,
        "max_steps": 20,
        "demo": True,
    }
)

# Load dataset (only 50 examples for quick demo)
dataset = load_dataset("trl-lib/Capybara", split="train[:50]")
print(f"✅ Dataset loaded: {len(dataset)} examples")
print(f"📝 Sample: {dataset[0]}")

# Training configuration
config = SFTConfig(
    # Hub settings - CRITICAL for saving results
    output_dir="qwen-demo-sft",
    push_to_hub=True,
    hub_model_id="evalstate/qwen-demo-sft",
    hub_strategy="end",  # Push only at end for demo
    
    # Training parameters (minimal for quick demo)
    max_steps=20,  # Very short training
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    
    # Logging
    logging_steps=5,
    save_strategy="no",  # Don't save checkpoints during training
    
    # Optimization
    warmup_steps=5,
    lr_scheduler_type="cosine",
    
    # Monitoring
    report_to="trackio",
)

# LoRA configuration (reduces memory usage)
peft_config = LoraConfig(
    r=8,  # Small rank for demo
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)

# Initialize trainer
print("🚀 Initializing trainer...")
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,
    args=config,
    peft_config=peft_config,
)

# Train
print("🔥 Starting training (20 steps)...")
trainer.train()

# Push to Hub
print("💾 Pushing model to Hub...")
trainer.push_to_hub()

# Finish Trackio tracking
trackio.finish()

print("✅ Training complete!")
print(f"📦 Model: https://huggingface.co/evalstate/qwen-demo-sft")
print(f"📊 Metrics: https://huggingface.co/spaces/evalstate/trackio-demo")

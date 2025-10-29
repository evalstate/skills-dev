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
    space_id="evalstate/demo-trackio-dashboard",
    config={
        "model": "Qwen/Qwen2.5-0.5B",
        "dataset": "trl-lib/Capybara",
        "examples": 50,
        "max_steps": 20,
        "note": "Quick demo training"
    }
)

# Load dataset (only 50 examples for quick demo)
dataset = load_dataset("trl-lib/Capybara", split="train[:50]")
print(f"✅ Dataset loaded: {len(dataset)} examples")

# Training configuration
config = SFTConfig(
    # Hub settings - CRITICAL for saving results
    output_dir="qwen-demo-sft",
    push_to_hub=True,
    hub_model_id="evalstate/qwen-demo-sft",
    
    # Quick training settings
    max_steps=20,  # Very short for demo
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    
    # Logging
    logging_steps=5,
    save_strategy="steps",
    save_steps=10,
    
    # Monitoring
    report_to="trackio",
)

# LoRA configuration (memory efficient)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)

# Initialize and train
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,
    args=config,
    peft_config=peft_config,
)

print("🚀 Starting demo training...")
trainer.train()

print("💾 Pushing to Hub...")
trainer.push_to_hub()

# Finish Trackio tracking
trackio.finish()

print("✅ Demo complete!")
print(f"📦 Model: https://huggingface.co/evalstate/qwen-demo-sft")
print(f"📊 Metrics: https://huggingface.co/spaces/evalstate/demo-trackio-dashboard")

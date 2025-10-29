# /// script
# dependencies = ["trl>=0.12.0", "peft>=0.7.0"]
# ///

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# Load 50 examples
dataset = load_dataset("trl-lib/Capybara", split="train[:50]")

# Train with minimal config
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,
    peft_config=LoraConfig(r=8, lora_alpha=16),
    args=SFTConfig(
        output_dir="output",
        push_to_hub=True,
        hub_model_id="evalstate/qwen-demo-minimal",
        max_steps=20,
        report_to="none",  # No monitoring for quick demo
    )
)

trainer.train()
trainer.push_to_hub()
print("✅ Done! Model at: https://huggingface.co/evalstate/qwen-demo-minimal")

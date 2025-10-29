# /// script
# dependencies = ["trl>=0.12.0", "peft>=0.7.0", "trackio"]
# ///

import os
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Training configuration with environment overrides"""
    MODEL = os.getenv("MODEL", "Qwen/Qwen2.5-0.5B")
    DATASET = os.getenv("DATASET", "trl-lib/Capybara")
    DATASET_SPLIT = os.getenv("DATASET_SPLIT", "train[:50]")
    OUTPUT_REPO = os.getenv("OUTPUT_REPO", "evalstate/qwen-demo")
    
    # Training
    MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))
    
    # LoRA
    LORA_R = int(os.getenv("LORA_R", "8"))
    LORA_ALPHA = int(os.getenv("LORA_ALPHA", "16"))
    
    # Monitoring
    TRACKIO_SPACE = os.getenv("TRACKIO_SPACE", "evalstate/ml-experiments")

# ============================================================================
# Setup Functions
# ============================================================================

def setup_monitoring(config: Config):
    """Initialize Trackio for experiment tracking"""
    import trackio
    
    project_name = config.OUTPUT_REPO.split('/')[-1]
    
    trackio.init(
        project=project_name,
        space_id=config.TRACKIO_SPACE,
        config={
            "model": config.MODEL,
            "dataset": config.DATASET,
            "max_steps": config.MAX_STEPS,
            "learning_rate": config.LEARNING_RATE,
            "lora_r": config.LORA_R,
        }
    )
    
    print(f"📊 Trackio: https://huggingface.co/spaces/{config.TRACKIO_SPACE}")

def load_and_validate_dataset(config: Config):
    """Load dataset and perform basic validation"""
    dataset = load_dataset(config.DATASET, split=config.DATASET_SPLIT)
    
    print(f"✅ Dataset loaded: {len(dataset)} examples")
    print(f"   Columns: {dataset.column_names}")
    
    # Basic validation
    assert len(dataset) > 0, "Dataset is empty!"
    assert "messages" in dataset.column_names, "Expected 'messages' column"
    
    return dataset

def create_training_config(config: Config) -> SFTConfig:
    """Create SFT training configuration"""
    return SFTConfig(
        # Output
        output_dir="output",
        push_to_hub=True,
        hub_model_id=config.OUTPUT_REPO,
        hub_strategy="end",
        
        # Training
        max_steps=config.MAX_STEPS,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=2,
        learning_rate=config.LEARNING_RATE,
        
        # Optimization
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        
        # Logging
        logging_steps=max(1, config.MAX_STEPS // 4),
        report_to="trackio",
        save_strategy="no",
    )

def create_peft_config(config: Config) -> LoraConfig:
    """Create LoRA/PEFT configuration"""
    return LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

# ============================================================================
# Main Training Flow
# ============================================================================

def main():
    """Main training pipeline"""
    config = Config()
    
    # Print configuration
    print("🚀 Training Configuration:")
    print(f"   Model: {config.MODEL}")
    print(f"   Dataset: {config.DATASET} ({config.DATASET_SPLIT})")
    print(f"   Output: {config.OUTPUT_REPO}")
    print(f"   Steps: {config.MAX_STEPS}")
    print(f"   Learning Rate: {config.LEARNING_RATE}")
    print(f"   LoRA r={config.LORA_R}, alpha={config.LORA_ALPHA}")
    print()
    
    # Setup
    setup_monitoring(config)
    dataset = load_and_validate_dataset(config)
    training_config = create_training_config(config)
    peft_config = create_peft_config(config)
    
    # Train
    print("🔥 Initializing trainer...")
    trainer = SFTTrainer(
        model=config.MODEL,
        train_dataset=dataset,
        args=training_config,
        peft_config=peft_config,
    )
    
    print("🏃 Training started...")
    trainer.train()
    
    print("💾 Pushing to Hub...")
    trainer.push_to_hub()
    
    # Cleanup
    import trackio
    trackio.finish()
    
    # Summary
    print()
    print("✅ Training Complete!")
    print(f"📦 Model: https://huggingface.co/{config.OUTPUT_REPO}")
    print(f"📊 Metrics: https://huggingface.co/spaces/{config.TRACKIO_SPACE}")

if __name__ == "__main__":
    main()

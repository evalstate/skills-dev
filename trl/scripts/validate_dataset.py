#!/usr/bin/env python3
# /// script
# dependencies = [
#     "datasets>=2.14.0",
# ]
# ///
"""
Validate dataset format for TRL training.

Usage:
    python validate_dataset.py <dataset_name> <method>
    
Examples:
    python validate_dataset.py trl-lib/Capybara sft
    python validate_dataset.py Anthropic/hh-rlhf dpo
"""

import sys
from datasets import load_dataset

def validate_sft_dataset(dataset):
    """Validate SFT dataset format."""
    print("🔍 Validating SFT dataset...")
    
    # Check for common fields
    columns = dataset.column_names
    print(f"📋 Columns: {columns}")
    
    has_messages = "messages" in columns
    has_text = "text" in columns
    
    if not (has_messages or has_text):
        print("❌ Dataset must have 'messages' or 'text' field")
        return False
    
    # Check first example
    example = dataset[0]
    
    if has_messages:
        messages = example["messages"]
        if not isinstance(messages, list):
            print("❌ 'messages' field must be a list")
            return False
        
        if len(messages) == 0:
            print("❌ 'messages' field is empty")
            return False
        
        # Check message format
        msg = messages[0]
        if not isinstance(msg, dict):
            print("❌ Messages must be dictionaries")
            return False
        
        if "role" not in msg or "content" not in msg:
            print("❌ Messages must have 'role' and 'content' keys")
            return False
        
        print("✅ Messages format valid")
        print(f"   First message: {msg['role']}: {msg['content'][:50]}...")
    
    if has_text:
        text = example["text"]
        if not isinstance(text, str):
            print("❌ 'text' field must be a string")
            return False
        
        if len(text) == 0:
            print("❌ 'text' field is empty")
            return False
        
        print("✅ Text format valid")
        print(f"   First text: {text[:100]}...")
    
    return True

def validate_dpo_dataset(dataset):
    """Validate DPO dataset format."""
    print("🔍 Validating DPO dataset...")
    
    columns = dataset.column_names
    print(f"📋 Columns: {columns}")
    
    required = ["prompt", "chosen", "rejected"]
    missing = [col for col in required if col not in columns]
    
    if missing:
        print(f"❌ Missing required fields: {missing}")
        return False
    
    # Check first example
    example = dataset[0]
    
    for field in required:
        value = example[field]
        if isinstance(value, str):
            if len(value) == 0:
                print(f"❌ '{field}' field is empty")
                return False
            print(f"✅ '{field}' format valid (string)")
        elif isinstance(value, list):
            if len(value) == 0:
                print(f"❌ '{field}' field is empty")
                return False
            print(f"✅ '{field}' format valid (list of messages)")
        else:
            print(f"❌ '{field}' must be string or list")
            return False
    
    return True

def validate_kto_dataset(dataset):
    """Validate KTO dataset format."""
    print("🔍 Validating KTO dataset...")
    
    columns = dataset.column_names
    print(f"📋 Columns: {columns}")
    
    required = ["prompt", "completion", "label"]
    missing = [col for col in required if col not in columns]
    
    if missing:
        print(f"❌ Missing required fields: {missing}")
        return False
    
    # Check first example
    example = dataset[0]
    
    if not isinstance(example["label"], bool):
        print("❌ 'label' field must be boolean")
        return False
    
    print("✅ KTO format valid")
    return True

def main():
    if len(sys.argv) != 3:
        print("Usage: python validate_dataset.py <dataset_name> <method>")
        print("Methods: sft, dpo, kto")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    method = sys.argv[2].lower()
    
    print(f"📦 Loading dataset: {dataset_name}")
    try:
        dataset = load_dataset(dataset_name, split="train")
        print(f"✅ Dataset loaded: {len(dataset)} examples")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        sys.exit(1)
    
    validators = {
        "sft": validate_sft_dataset,
        "dpo": validate_dpo_dataset,
        "kto": validate_kto_dataset,
    }
    
    if method not in validators:
        print(f"❌ Unknown method: {method}")
        print(f"Supported methods: {list(validators.keys())}")
        sys.exit(1)
    
    validator = validators[method]
    valid = validator(dataset)
    
    if valid:
        print(f"\n✅ Dataset is valid for {method.upper()} training")
        sys.exit(0)
    else:
        print(f"\n❌ Dataset is NOT valid for {method.upper()} training")
        sys.exit(1)

if __name__ == "__main__":
    main()

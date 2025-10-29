# Trackio Integration for TRL Training

**Trackio** is an experiment tracking library that provides real-time metrics visualization for remote training on Hugging Face Jobs infrastructure.

⚠️ **IMPORTANT**: For Jobs training (remote cloud GPUs):
- Training happens on ephemeral cloud runners (not your local machine)
- Trackio syncs metrics to a Hugging Face Space for real-time monitoring
- Without a Space, metrics are lost when the job completes
- The Space dashboard persists your training metrics permanently

## Setting Up Trackio for Jobs

**Step 1: Add trackio dependency**
```python
# /// script
# dependencies = [
#     "trl>=0.12.0",
#     "trackio",  # Required!
# ]
# ///
```

**Step 2: Create a Trackio Space (one-time setup)**

**Option A: Let Trackio auto-create (Recommended)**
Pass a `space_id` to `trackio.init()` and Trackio will automatically create the Space if it doesn't exist.

**Option B: Create manually**
- Create Space via Hub UI at https://huggingface.co/new-space
- Select Gradio SDK
- OR use command: `huggingface-cli repo create my-trackio-dashboard --type space --space_sdk gradio`

**Step 3: Initialize Trackio with space_id**
```python
import trackio

trackio.init(
    project="my-training",
    space_id="username/trackio",  # CRITICAL for Jobs! Replace 'username' with your HF username
    config={
        "model": "Qwen/Qwen2.5-0.5B",
        "dataset": "trl-lib/Capybara",
        "learning_rate": 2e-5,
    }
)
```

**Step 4: Configure TRL to use Trackio**
```python
SFTConfig(
    report_to="trackio",
    # ... other config
)
```

**Step 5: Finish tracking**
```python
trainer.train()
trackio.finish()  # Ensures final metrics are synced
```

## What Trackio Tracks

Trackio automatically logs:
- ✅ Training loss
- ✅ Learning rate
- ✅ GPU utilization
- ✅ Memory usage
- ✅ Training throughput
- ✅ Custom metrics

## How It Works with Jobs

1. **Training runs** → Metrics logged to local SQLite DB
2. **Every 5 minutes** → Trackio syncs DB to HF Dataset (Parquet)
3. **Space dashboard** → Reads from Dataset, displays metrics in real-time
4. **Job completes** → Final sync ensures all metrics persisted

## Viewing the Dashboard

After starting training:
1. Navigate to the Space: `https://huggingface.co/spaces/username/trackio`
2. The Gradio dashboard shows all tracked experiments
3. Filter by project, compare runs, view charts with smoothing

## Recommendation

- **Trackio**: Best for real-time monitoring during long training runs
- **Weights & Biases**: Best for team collaboration, requires account

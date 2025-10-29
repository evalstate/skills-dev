from huggingface_hub import run_uv_job

job = run_uv_job(
    script="https://huggingface.co/evalstate/demo-training-scripts/resolve/main/train_demo.py",
    flavor="t4-small",
    timeout="20m",
    secrets={"HF_TOKEN": "$HF_TOKEN"},
)

print(f"\n✅ Job submitted successfully!")
print(f"Job ID: {job.job_id}")
print(f"Monitor: https://huggingface.co/jobs/{job.job_id}")
print(f"\nExpected time: ~10-15 minutes")
print(f"Estimated cost: ~$0.20")
print(f"\nThe job is running in the background!")
print(f"📊 Once training starts, view metrics at: https://huggingface.co/spaces/evalstate/training-demo-dashboard")

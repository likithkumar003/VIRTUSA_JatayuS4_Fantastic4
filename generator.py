# synthetic_agentic/generator.py

def train_and_generate(model, data, num_rows: int):
    """Train the selected model and generate synthetic data."""
    try:
        print("🚀 Training model...")
        model.fit(data)
        print("✅ Model trained. Generating synthetic data...")
        synthetic_data = model.sample(num_rows)
        print("✅ Synthetic data generated.")
        return synthetic_data
    except Exception as e:
        print(f"[✘] Generation failed: {e}")
        raise

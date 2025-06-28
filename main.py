from data_loader import load_data
from metadata_generator import generate_metadata
from model_selector_agent import auto_select_model
from generator import train_and_generate
from evaluation_agent import evaluate_quality
from feedback_loop_agent import feedback_loop
from utility_evaluator_agent import evaluate_utility

REAL_DATA_PATH = "sample_classification_health.csv"
SYNTHETIC_CSV = "synthetic_output.csv"
QUALITY_THRESHOLD = 0.90

# Step 1: Load Data
data = load_data(REAL_DATA_PATH)

# Step 2: Generate Metadata
metadata = generate_metadata(data)

# Step 3: Select Model
# model = select_model(data, metadata)
from model_selector_agent import auto_select_model
model = auto_select_model(data, metadata)

# Step 4–6: Feedback loop handles training, generation & quality control
# synthetic_data = feedback_loop(model, data, metadata, threshold=QUALITY_THRESHOLD)

num_rows = int(input(f"\n📏 Enter number of synthetic rows to generate (original has {len(data)}): "))

synthetic_data = feedback_loop(
    model=model,
    data=data,
    metadata=metadata,
    threshold=QUALITY_THRESHOLD,
    num_rows=num_rows
)


# Step 7: Utility Evaluation (optional, but agentic)
target_col = "is_irregular"
utility_score = evaluate_utility(data, synthetic_data, target_col)
import matplotlib.pyplot as plt

def plot_accuracy(real_acc, synthetic_acc):
    plt.figure(figsize=(6, 4))
    bars = plt.bar(["Real", "Synthetic"], [real_acc, synthetic_acc], color=["green", "orange"])
    plt.title("📊 Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.bar_label(bars, fmt="%.2f", label_type="edge", padding=5)
    plt.tight_layout()
    plt.show()

plot_accuracy(real_acc=utility_score, synthetic_acc=utility_score)


# Step 8: Save Output
synthetic_data.to_csv(SYNTHETIC_CSV, index=False)
print(f"\n[✔] Final synthetic data saved to: {SYNTHETIC_CSV}")

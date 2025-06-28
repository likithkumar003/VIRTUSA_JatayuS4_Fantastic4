# synthetic_agentic/feedback_loop_agent.py

from generator import train_and_generate
from evaluation_agent import evaluate_quality


def feedback_loop(model, data, metadata, threshold=0.9, num_rows=1000):
    attempt = 1
    best_score = 0
    best_data = None

    while attempt <= 3:
        print(f"\n🔁 [Attempt {attempt}] Training and generating synthetic data...")
        model.fit(data)
        synthetic_data = model.sample(num_rows)

        score = evaluate_quality(data, synthetic_data, metadata)

        if score >= threshold:
            print(f"[✔] Passed quality threshold with score: {score:.2f}")
            return synthetic_data,score

        print(f"[✘] Quality too low ({score:.2f}), retrying...\n")
        if score > best_score:
            best_score = score
            best_data = synthetic_data

        attempt += 1

    print(f"[⚠] Returning best attempt (score: {best_score:.2f})")
    return best_data, best_score

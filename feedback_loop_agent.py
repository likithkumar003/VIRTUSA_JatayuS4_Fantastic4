from generator import train_and_generate
from evaluation_agent import evaluate_quality
from utility_evaluator_agent import evaluate_utility
from model_selector_agent import auto_select_model
import pandas as pd

def agentic_feedback_loop(data, metadata, target_col, threshold=0.90, num_rows=1000):
    # from sdv.tabular import CTGAN, TVAE, GaussianCopula
    from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer


    models = {
        "CTGAN": CTGANSynthesizer(metadata),
        "TVAE": TVAESynthesizer(metadata),
        "GaussianCopula": GaussianCopulaSynthesizer(metadata)
    }

    best_combo = None
    best_score = 0

    for name, model in models.items():
        print(f"\n🤖 Trying Model: {name}")

        attempt = 1
        best_model_score = 0
        best_model_data = None

        while attempt <= 3:
            print(f"🔁 Attempt {attempt} for {name}")
            model.fit(data)
            synthetic_data = model.sample(num_rows)

            score = evaluate_quality(data, synthetic_data, metadata)
            utility = evaluate_utility(data, synthetic_data, target_col)

            combo_score = (score + utility) / 2

            print(f"📝 Quality: {score:.2f} | Utility: {utility:.2f} | Combined: {combo_score:.2f}")

            if score >= threshold and combo_score > best_score:
                best_combo = (synthetic_data, score, utility, name)
                best_score = combo_score
                print(f"✅ Acceptable — Saving this combination: {name}")
                break

            if combo_score > best_model_score:
                best_model_score = combo_score
                best_model_data = (synthetic_data, score, utility, name)

            attempt += 1

        if best_combo is None and best_model_score > best_score:
            best_combo = best_model_data
            best_score = best_model_score

    if best_combo:
        print(f"\n[✔] Best combo: {best_combo[3]} | Quality: {best_combo[1]:.2f} | Utility: {best_combo[2]:.2f}")
        return best_combo
    else:
        raise ValueError("No valid synthetic data passed the agentic loop.")

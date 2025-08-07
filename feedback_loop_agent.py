from generator import train_and_generate
from evaluation_agent import evaluate_quality
from utility_evaluator_agent import evaluate_utility
from model_selector_agent import auto_select_model
import pandas as pd
import streamlit as st
import time

def agentic_feedback_loop(data, metadata, target_col, threshold=0.90, num_rows=1000):
    from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer

    models = {
        "CTGAN": CTGANSynthesizer(metadata),
        "TVAE": TVAESynthesizer(metadata),
        "GaussianCopula": GaussianCopulaSynthesizer(metadata)
    }

    best_combo = None
    best_score = 0
    attempt_history = []

    total_models = len(models)
    current_model_index = 0

    # ✅ Create placeholders
    status_placeholder = st.empty()
    progress_placeholder = st.empty()

    for name, model in models.items():
        current_model_index += 1
        progress = int((current_model_index - 1) / total_models * 100)
        print(f"\n🤖 Trying Model: {name}")
        status_placeholder.info(f"🤖 Trying Model: **{name}**")
        progress_placeholder.progress(progress)

        attempt = 1
        best_model_score = 0
        best_model_data = None

        while attempt <= 3:
            print(f"🔁 Attempt {attempt} for {name}")
            status_placeholder.write(f"🔁 Attempt **{attempt}** for **{name}** ...")
            # ⏳ Small sleep to visually show (optional)
            # time.sleep(0.5)

            model.fit(data)
            synthetic_data = model.sample(num_rows)

            score = evaluate_quality(data, synthetic_data, metadata)
            utility = evaluate_utility(data, synthetic_data, target_col)
            combo_score = (score + utility) / 2

            attempt_history.append({
                "attempt": attempt,
                "model": name,
                "quality": score,
                "utility": utility,
                "combined": combo_score
            })

            status_placeholder.write(
                f"📝 Quality: **{score:.2f}** | Utility: **{utility:.2f}** | Combined: **{combo_score:.2f}**"
            )

            if score >= threshold and combo_score > best_score:
                best_combo = (synthetic_data, score, utility, name)
                best_score = combo_score
                status_placeholder.success(f"✅ Saved combination: {name} ✔️")
                break

            if combo_score > best_model_score:
                best_model_score = combo_score
                best_model_data = (synthetic_data, score, utility, name)

            attempt += 1

        # Update progress
        progress = int(current_model_index / total_models * 100)
        progress_placeholder.progress(progress)

        if best_combo is None and best_model_score > best_score:
            best_combo = best_model_data
            best_score = best_model_score

    progress_placeholder.empty()
    status_placeholder.success(f"✅ Best combo: **{best_combo[3]}** | Quality: **{best_combo[1]:.2f}** | Utility: **{best_combo[2]:.2f}**")

    if best_combo:
        synthetic_data, score, utility, name = best_combo
        return synthetic_data, score, utility, name, attempt_history
    else:
        raise ValueError("No valid synthetic data passed the agentic loop.")

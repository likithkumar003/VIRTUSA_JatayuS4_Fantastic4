# row_predictor_agent.py

def predict_row_count(real_data, selected_model_name):
    """
    Simple agent logic to predict recommended number of synthetic rows
    based on input data size and selected model type.
    """
    real_rows = len(real_data)

    # Base factor by dataset size
    if real_rows <= 500:
        multiplier = 3.7  # small data: boost
    elif real_rows <= 2000:
        multiplier = 2.5  # medium data: moderate boost
    else:
        multiplier = 1.8  # large data: slight expansion

    # Adjust factor by model type
    if "CTGAN" in selected_model_name:
        multiplier *= 1.2  # CTGAN good for big upsizing
    elif "TVAE" in selected_model_name:
        multiplier *= 1.1  # TVAE moderate
    elif "GaussianCopula" in selected_model_name:
        multiplier *= 1.0  # GaussianCopula: stable, less boost

    # Final calculation
    suggested_rows = int(real_rows * multiplier)
    return suggested_rows

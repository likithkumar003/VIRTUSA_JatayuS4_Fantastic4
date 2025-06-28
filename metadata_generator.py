# synthetic_agentic/metadata_generator.py

from sdv.metadata import SingleTableMetadata
import pandas as pd

def generate_metadata(data: pd.DataFrame) -> SingleTableMetadata:
    """Generate SDV metadata from a DataFrame."""
    try:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)
        print("[✔] Metadata generated.")
        return metadata
    except Exception as e:
        print(f"[✘] Failed to generate metadata: {e}")
        raise

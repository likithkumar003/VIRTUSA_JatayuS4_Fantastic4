import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.lite import SingleTablePreset
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/lifesat/oecd_bli_2015.csv"
data = pd.read_csv(url)

# Define metadata from the dataset
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)

# Initialize the synthesizer with metadata
synthesizer = SingleTablePreset(metadata=metadata, name='FAST_ML')
synthesizer.fit(data)

# Generate synthetic data
synthetic_data = synthesizer.sample(1000)
print("\nSynthetic Data:")
print(synthetic_data.head())

# Save synthetic data to file
synthetic_data.to_csv("synthetic_data.csv", index=False)

# ---- Visualization ----

# Separate numeric and categorical columns
numeric_columns = data.select_dtypes(include='number').columns
categorical_columns = data.select_dtypes(exclude='number').columns

# ✅ Plot numeric columns (KDE plots)
for column in numeric_columns:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data[column], label="Real Data", fill=True, alpha=0.5)
    sns.kdeplot(synthetic_data[column], label="Synthetic Data", fill=True, alpha=0.5)
    plt.legend()
    plt.title(f"Distribution of {column}")
    plt.show()

# ✅ Plot categorical columns (Bar plots)
for column in categorical_columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=column, data=data, color='blue', label="Real Data", alpha=0.5)
    sns.countplot(x=column, data=synthetic_data, color='orange', label="Synthetic Data", alpha=0.5)
    plt.legend()
    plt.title(f"Category Distribution: {column}")
    plt.xticks(rotation=45)
    plt.show()

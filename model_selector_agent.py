from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer

def auto_select_model(data, metadata):
    num_num = len(data.select_dtypes(include='number').columns)
    num_cat = len(data.select_dtypes(exclude='number').columns)

    print(f"\n🤖 Auto Model Selector: {num_num} numeric / {num_cat} categorical")

    if num_cat > num_num:
        print("🧠 Selected: CTGAN (better for categorical-heavy data)")
        return CTGANSynthesizer(metadata)
    elif num_num > num_cat:
        print("🧠 Selected: GaussianCopula (better for numeric-heavy data)")
        return GaussianCopulaSynthesizer(metadata)
    else:
        print("🧠 Selected: TVAE (balanced)")
        return TVAESynthesizer(metadata)

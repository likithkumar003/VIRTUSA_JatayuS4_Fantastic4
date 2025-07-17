from langchain_google_genai import ChatGoogleGenerativeAI

def predict_target_column(df, sample_rows=50):
    cols = list(df.columns)
    rows = df.head(sample_rows).to_dict(orient="records")

    prompt = f"""
    You are an expert data analyst. Your job is to guess the target column (the variable that is likely the outcome, label, or dependent variable) for this dataset.

    Below are the columns and a few sample rows. Analyze them and answer ONLY with the name of the most likely target column.

    Columns: {cols}
    Sample Rows: {rows}

    Rules:
    - The target column is usually something that can be predicted or explained by the other columns.
    - The target column is the variable explained or predicted by the other columns.
    - Consider the logical flow: the target is usually an effect, the others are causes.
    - That can influence others to make predictions. make sure that influence every column.
    - If there is a column like “target”, “label”, “output”, “result”, or “class”, prefer that.
    - If there is a clear ID column (like “id”, “user_id”), never pick that.
    - If you are unsure, pick the column that looks like an outcome variable, not an input feature.
    - Reply with ONLY the column name and nothing else.
    """

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    result = llm.invoke(prompt)
    return result.content.strip()

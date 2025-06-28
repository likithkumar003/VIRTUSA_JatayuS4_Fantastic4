

import streamlit as st
import pandas as pd
from data_loader import load_data
from metadata_generator import generate_metadata
from model_selector_agent import auto_select_model
from feedback_loop_agent import feedback_loop
from utility_evaluator_agent import evaluate_utility
from rag_chatbot import launch_rag_interface

st.set_page_config(page_title="SmartSynth with RAG", layout="wide")

# ───────────────────── Title & Mode Select ─────────────────────
st.title("SmartSynth + RAG Chatbot: Agentic Tabular Data Generator with Rag chatbot Assistant")

# Session vars
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

    # st.markdown("## 🧠 Mode")
chatbot_mode = st.sidebar.selectbox("Select Chatbot Mode", ["Use Uploaded PDFs", "Use Synthetic CSV", "Use PDFs and CSVs"])

# ─────────────── Sidebar Upload Logic ───────────────
with st.sidebar:
    st.header("📚For RAG Assistant")

    if chatbot_mode == "Use Uploaded PDFs":
        pdf_docs = st.file_uploader("📄 Upload PDF files", type=["pdf"], accept_multiple_files=True)
        if st.button("📖 Process PDFs"):
            if pdf_docs:
                launch_rag_interface(pdf_docs=pdf_docs)

    elif chatbot_mode == "Use Synthetic CSV":
        uploaded_csv = st.file_uploader("📎 Upload synthetic_output.csv", type=["csv"])
        if uploaded_csv:
            with open("synthetic_output.csv", "wb") as f:
                f.write(uploaded_csv.read())
            st.success("✅ synthetic_output.csv uploaded!")
            launch_rag_interface(csv_path="synthetic_output.csv")
        else:
            st.info("💡 Run SmartSynth on the right ➜ Download CSV ➜ Re-upload here to chat with it.")

    elif chatbot_mode == "Use PDFs and CSVs":
        pdf_docs = st.file_uploader("📄 Upload PDFs", type=["pdf"], accept_multiple_files=True, key="hybrid_pdfs")
        csv_docs = st.file_uploader("📊 Upload CSV files", type=["csv"], accept_multiple_files=True, key="hybrid_csvs")
        if st.button("🔁 Process Hybrid"):
            if pdf_docs or csv_docs:
                launch_rag_interface(pdf_docs=pdf_docs, csv_docs=csv_docs)
            else:
                st.warning("Please upload at least one PDF or CSV file.")


# ───────────────────── Layout Split ─────────────────────
col1, col2 = st.columns([1, 1])

# ─────────────── Left Panel: RAG Interaction ───────────────
with col2:
    st.header("🤖 RAG Chatbot Assistant")


    question = st.text_input("🔍 Ask your question", key="chat_input", placeholder="Type your question and hit enter...")

    if question:
        from rag_chatbot import get_rag_answer
        response = get_rag_answer(question)
        st.session_state.chat_history.append((question, response))
        st.markdown(f"**🧠 Question:** {question}")
        st.markdown(f"**📘 Answer:** {response}")
        # st.session_state["chat_input"] = ""


    # Download chat log
    if st.session_state.chat_history:
        full_chat = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history])
        st.download_button("💾 Download Chat Log", full_chat, file_name="chat_log.txt")

# ─────────────── Right Panel: SmartSynth ───────────────
with col1:
    st.header("🧬 SmartSynth Generation")

    uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"], key="main_csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head())

        target_col = st.selectbox("🎯 Select target column", df.columns)
        num_rows = st.number_input("📏 Number of synthetic rows", min_value=10, max_value=10000, value=len(df), step=100)

        if st.button("⚡ Generate Synthetic Data"):
            metadata = generate_metadata(df)
            model = auto_select_model(df, metadata)
            synthetic_df, quality_score, model_used, utility_score = feedback_loop(
                model=model,
                data=df,
                metadata=metadata,
                threshold=0.90,
                num_rows=num_rows
            )

            st.success("✅ Synthetic data generated!")
            st.info(f"📊 Quality Score: {quality_score:.2f}")
            real_acc, synth_acc = evaluate_utility(df, synthetic_df, target_col, return_both=True)
            st.write(f"📊 Real Accuracy: {real_acc:.2f} | Synthetic Accuracy: {synth_acc:.2f}")
            st.download_button("⬇️ Download Synthetic CSV", synthetic_df.to_csv(index=False), file_name="synthetic_output.csv")

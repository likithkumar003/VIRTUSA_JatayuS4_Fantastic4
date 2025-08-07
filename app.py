import streamlit as st
import pandas as pd
from data_loader import load_data
from metadata_generator import generate_metadata
from model_selector_agent import auto_select_model
from utility_evaluator_agent import evaluate_utility
from rag_chatbot import launch_rag_interface, get_rag_answer, structured_query_agent, fallback_gemini
from row_predictor_agent import predict_row_count
import plotly.express as px
import plotly.graph_objects as go
import time



st.set_page_config(page_title="SmartSynth with RAG", layout="wide")

st.title("SmartSynth + RAG: Agentic Tabular Data Generator with Rag Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

chatbot_mode = st.sidebar.selectbox(
    "Chatbot Mode",
    ["Use Uploaded PDFs", "Use Synthetic CSV", "Use PDFs and CSVs"]
)

with st.sidebar:
    st.header("📚For RAG Setup")
    if chatbot_mode == "Use Uploaded PDFs":
        pdf_docs = st.file_uploader("📄 Upload PDFs", type=["pdf"], accept_multiple_files=True)
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
            st.info("💡 Run SmartSynth → Download CSV → Re-upload here.")
    elif chatbot_mode == "Use PDFs and CSVs":
        pdf_docs = st.file_uploader("📄 Upload PDFs", type=["pdf"], accept_multiple_files=True, key="hybrid_pdfs")
        csv_docs = st.file_uploader("📊 Upload CSVs", type=["csv"], accept_multiple_files=True, key="hybrid_csvs")
        if st.button("🔁 Process Hybrid"):
            if pdf_docs or csv_docs:
                launch_rag_interface(pdf_docs=pdf_docs, csv_docs=csv_docs)

col1, col2 = st.columns([1, 1])

with col2:
    st.header("🤖 RAG Chatbot Assistant")
    question = st.text_input(
        "🔍 Ask your question",
        key="chat_input",
        placeholder="Ask about your data..."
    )

    if question:
        if chatbot_mode == "Use Uploaded PDFs":
            response = get_rag_answer(question)
        elif chatbot_mode == "Use Synthetic CSV":
            response = structured_query_agent(question, ["synthetic_output.csv"])
        elif chatbot_mode == "Use PDFs and CSVs":
            response = fallback_gemini(question)
        else:
            response = fallback_gemini(question)

        st.session_state.chat_history.append((question, response))
        st.markdown(f"**🧠 Question:** {question}")
        st.markdown(f"**📘 Answer:** {response}")

    if st.session_state.chat_history:
        chat_log = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history])
        st.download_button("💾 Download Chat Log", chat_log, file_name="chat_log.txt")

with col1:
    st.header("🧬 SmartSynth Generation")

    uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"], key="main_csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head())

        import target_predictor_agent as tpa
        auto_target = tpa.predict_target_column(df)
        st.info(f"🤖Agent Suggested target: {auto_target}")
        target_col = st.selectbox("🎯 Select target", df.columns, index=df.columns.get_loc(auto_target))

        metadata = generate_metadata(df)
        model1 = auto_select_model(df, metadata)
        suggested_rows = predict_row_count(df, model1.__class__.__name__)
        st.info(f"🤖Agent Suggested rows: **{suggested_rows}**")
        num_rows = st.number_input("📏 Synthetic rows", 10, 100000, suggested_rows, step=100)

        if st.button("⚡ Generate Synthetic Data"):
            with st.spinner("🔍 Analyzing your data and metadata..."):
                time.sleep(2)  # Simulate

            

            # 🎬 Animated progress steps
            my_progress = st.progress(0, text="🚀 Starting agentic feedback loop...")

            steps = [
        "🔢 Choosing candidate models...",
        "🤖 Running multiple attempts...",
        "🔬 Evaluating quality & utility...",
        "🔁 Selecting best combination...",
        "✅ Finalizing synthetic data..."
            ]

            for i, step in enumerate(steps):
                my_progress.progress((i + 1) * 20, text=step)
                time.sleep(1.2)  # simulate step time

            from feedback_loop_agent import agentic_feedback_loop
            synthetic_df, quality_score, utility_score, model_used, attempt_history= agentic_feedback_loop(
                data=df,
                metadata=metadata,
                target_col=target_col,
                threshold=0.90,
                num_rows=num_rows
            )
            st.info(f"📊 Quality Score: {quality_score:.2f}")
            real_acc, synth_acc = evaluate_utility(df, synthetic_df, target_col, return_both=True)
            st.write(f"📊 Real: {real_acc:.2f} | Synthetic: {synth_acc:.2f}")
            # 📊 Visualize attempts
            attempts_df = pd.DataFrame(attempt_history)
            if not attempts_df.empty:
                fig_bar = px.bar(
            attempts_df.melt(id_vars=["attempt", "model"], value_vars=["quality", "utility"]),
            x="attempt",
            y="value",
            color="variable",
            barmode="group",
            title="Attempts: Quality vs Utility",
            hover_data=["model"]
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=quality_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Final Quality Score — {model_used}"},
            gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 0.5], 'color': "lightgray"},
                {'range': [0.5, 0.8], 'color': "yellow"},
                {'range': [0.8, 1], 'color': "limegreen"}
            ],
            }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.success("✅ Synthetic data generated! You can download👇😉")
            st.download_button("⬇️ Download CSV", synthetic_df.to_csv(index=False), file_name="synthetic_output.csv")

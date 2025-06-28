🧠 SmartSynth: Agentic Tabular Data Generation with Adaptive Model Selection
SmartSynth is an intelligent system that automatically generates high-quality, utility-preserving synthetic tabular data. It uses agentic AI to select the best model, improve generation iteratively, and integrates a powerful RAG-based chatbot using Gemini 1.5 for querying uploaded PDFs or synthetic data.

🚀 Features
🌱 Stage 2: Agentic Synthetic Data Generator
📊 Auto Model Selector
Picks the best-fit model (CTGAN, TVAE, or GaussianCopula) based on data characteristics.

♻️ Feedback Loop Agent
Trains, evaluates, and re-generates synthetic data until quality threshold is met (via SDMetrics).

📈 Utility Evaluator
Measures downstream ML performance (e.g., classifier accuracy) on synthetic vs real data.

📦 Streamlit UI v1
Upload CSV, select target column, control output rows, generate and download data.

🔥 Stage 3: RAG-Powered Gemini Integration
📚 RAG Chatbot (Sidebar)
Ask questions from uploaded PDFs, synthetic CSVs, or both – powered by Gemini 1.5 Flash.

🔄 Hybrid Upload Support
Supports PDF + CSV at the same time, builds unified knowledge base with FAISS.

💬 Context-Aware QA
Retrieves relevant chunks from data, answers questions, and falls back to Gemini if context is missing.

🧠 Intelligent Memory
Stores chat logs in session; you can download them at any time.

📐 Clean UI
Split-panel layout: SmartSynth on the right, RAG chatbot in the sidebar.

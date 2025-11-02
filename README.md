# 🧠 SmartSynth

**Agentic Tabular Data Generation + RAG-Powered Q&A Assistant**

SmartSynth is an advanced synthetic data generation system with **agentic AI** for feedback-based model selection **and** an integrated **Gemini RAG chatbot** for interactive question-answering on uploaded PDFs **and** generated CSVs.  
Built for privacy-preserving data creation, analysis, and domain-specific exploration.

---

## 🎯 What it Does

- ✅ Automatically picks the best synthetic model (**CTGAN**, **TVAE**, or **GaussianCopula**) based on your dataset’s structure.
- ✅ Uses an **agentic feedback loop** to ensure the generated synthetic data meets your chosen quality threshold.
- ✅ Compares real vs synthetic utility scores using scikit-learn.
- ✅ Lets you **ask questions** about your uploaded PDFs **or** the generated CSVs with a **RAG-powered** Gemini 1.5 chatbot.
- ✅ Supports **hybrid mode** — combine PDFs and CSVs for a unified knowledge base.
- ✅ Download synthetic data **and** your chat logs for auditing.

  ## 🎯 How it Does

  <img width="1920" height="1080" alt="Company Overview" src="https://github.com/user-attachments/assets/f1c9ae25-610f-4d88-819b-4225eb2719ae" />



---

## 🚀 Hackathon Context

| 📌 | Detail |
|----|--------|
| 🏆 Hackathon | Virtusa Jatayu Season 4 |
| 📅 Stage 2 | Core Agentic AI + Feedback Loop |
| ✅ Stage 3 | Enhanced with Gemini RAG Q&A |
| ⏰ Deadline | Up to July 31 |

---

## ⚙️ Tech Stack

**Core:**
- **Python**
- **Streamlit** — Interactive web UI

**Synthetic Data:**
- **SDV** — CTGAN, TVAE, GaussianCopula
- **SDMetrics** — Quality evaluation
- **pandas**, **scikit-learn**

**RAG Chatbot:**
- **LangChain**
- **FAISS** — Local vector store
- **Google Generative AI API** — Gemini 1.5 Flash
- **PyPDF2** — PDF extraction

**Other:**
- `.env` for secrets
- Git for versioning

---

## 📂 Project Structure

├── app.py # Main Streamlit app: SmartSynth + RAG sidebar

├── rag_chatbot.py # RAG logic: PDF + CSV + hybrid support

├── data_loader.py # CSV loader

├── metadata_generator.py # Metadata inference

├── model_selector_agent.py # Adaptive model selector

├── feedback_loop_agent.py # Agentic feedback loop

├── utility_evaluator_agent.py # Utility comparison logic

├── requirements.txt

├── .env # Store GOOGLE_API_KEY


---

## 🗂️ Key Features

| Feature | Stage |
|---------|-------|
| Adaptive model selection | ✅ Stage 2 |
| Agentic feedback loop | ✅ Stage 2 |
| Quality & Utility scoring | ✅ Stage 2 |
| Gemini RAG PDF/CSV Q&A | ✅ Stage 3 |
| Hybrid PDF + CSV RAG | ✅ Stage 3 |
| Session-based chat logs | ✅ Stage 3 |
| Download chat logs | ✅ Stage 3 |

---

## 🏃 How to Run Locally

1️⃣ **Clone & Setup**

    git clone "https://github.com/likithkumar003/VIRTUSA_JatayuS4_Fantastic4"
    cd VIRTUSA_JatayuS4_Fantastic4

    
2️⃣**Activate Environment:**

    python -m venv venv
    venv\Scripts\activate  # Windows
    source venv/bin/activate  # macOS/Linux


3️⃣**Install Dependencies:**

    pip install -r Requirements.txt
    

4️⃣**Make sure add your API key in .env file**

    GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY

5️⃣**Run**

    streamlit run app.py


# import streamlit as st
# import os
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# import pandas as pd

# # Load Gemini key
# from dotenv import load_dotenv
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# # PDF utility
# def extract_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         reader = PdfReader(pdf)
#         for page in reader.pages:
#             content = page.extract_text()
#             if content:
#                 text += content
#     return text

# # CSV utility
# def extract_csv_text(csv_path):
#     df = pd.read_csv(csv_path)
#     return df.to_string(index=False)

# def chunk_text(text):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     return splitter.split_text(text)

# def vector_store(chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     store = FAISS.from_texts(chunks, embedding=embeddings)
#     store.save_local("faiss_index")

# def get_qa_chain():
#     prompt_template = """
#     Use the context below to answer the question. 
#     If not found, say "Answer not available in the context."

#     Context:
#     {context}

#     Question:
#     {question}

#     Answer:
#     """
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
#     return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

# def ask_question(query):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = db.similarity_search(query)
#     chain = get_qa_chain()
#     response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
#     st.write("📄 Answer:", response["output_text"])

# def launch_rag_interface(pdf_docs=None, csv_path=None):
#     if pdf_docs:
#         with st.spinner("🔍 Reading PDFs..."):
#             text = extract_pdf_text(pdf_docs)
#     elif csv_path:
#         with st.spinner("📊 Reading CSV..."):
#             text = extract_csv_text(csv_path)
#     else:
#         st.warning("⚠ No data source provided.")
#         return

#     chunks = chunk_text(text)
#     vector_store(chunks)
#     st.success("✅ Data processed successfully!")

#     st.text_input("🧠 Ask a question from your document:", on_change=lambda: ask_question(st.session_state.query), key="query")


# rag_chatbot.py






















# import streamlit as st
# from PyPDF2 import PdfReader
# import pandas as pd
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import os

# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=GOOGLE_API_KEY)


# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         reader = PdfReader(pdf)
#         for page in reader.pages:
#             if page.extract_text():
#                 text += page.extract_text()
#     return text


# def get_csv_text(csv_docs):
#     combined_text = ""
#     for csv_file in csv_docs:
#         df = pd.read_csv(csv_file)
#         combined_text += df.to_csv(index=False) + "\n"
#     return combined_text


# def get_text_chunks(text):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     return splitter.split_text(text)


# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")


# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context.
#     If the answer is not in the context, just say "answer is not available in the context".
#     Do not make up an answer.

#     Context: {context}
#     Question: {question}
#     Answer:
#     """
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
#     return load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)


# def user_input_qa(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     return response["output_text"]


# def launch_rag_interface(pdf_docs=None, csv_docs=None):
#     st.subheader("💬 Ask Your Question")

#     if pdf_docs:
#         raw_text = get_pdf_text(pdf_docs)
#     elif csv_docs:
#         raw_text = get_csv_text(csv_docs)
#     else:
#         st.warning("❌ Please upload file(s) first.")
#         return

#     chunks = get_text_chunks(raw_text)
#     get_vector_store(chunks)

#     if "rag_chat_history" not in st.session_state:
#         st.session_state.rag_chat_history = []

#     user_q = st.text_input("🔍 Ask a question about your data:", key="rag_input")
#     if user_q:
#         answer = user_input_qa(user_q)
#         st.session_state.rag_chat_history.append((user_q, answer))

#     for q, a in reversed(st.session_state.rag_chat_history):
#         st.markdown(f"**❓ {q}**")
#         st.markdown(f"**🧠 {a}**")



# import streamlit as st
# from PyPDF2 import PdfReader
# import pandas as pd
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import os

# # Setup
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=GOOGLE_API_KEY)

# def get_pdf_text(pdf_docs):
#     texts = [
#         PdfReader(pdf).pages[i].extract_text()
#         for pdf in pdf_docs for i in range(len(PdfReader(pdf).pages))
#     ]
#     return "\n".join(filter(None, texts))

# def get_csv_texts(csv_docs):
#     return [pd.read_csv(f).to_csv(index=False) for f in csv_docs]

# def split_to_chunks(text):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     return splitter.split_text(text)

# def build_vector_store(texts):
#     chunks = sum((split_to_chunks(t) for t in texts), [])
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     store = FAISS.from_texts(chunks, embedding=embeddings)
#     store.save_local("faiss_index")

# def get_chain():
#     prompt = PromptTemplate(
#         template="""
#         Answer the question from the context. If not present, say "answer is not available in context."
#         Context: {context}
#         Question: {question}
#         """,
#         input_variables=["context", "question"],
#     )
#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
#     return load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

# def launch_rag_interface(pdf_docs=None, csv_docs=None):
#     st.subheader("🤖 Chat Assistant (RAG + Gemini Fallback)")
#     raw_texts = []
#     if pdf_docs:
#         st.write(f"📌 Loaded {len(pdf_docs)} PDF file(s).")
#         raw_texts.append(get_pdf_text(pdf_docs))
#     if csv_docs:
#         st.write(f"📌 Loaded {len(csv_docs)} CSV file(s).")
#         raw_texts.extend(get_csv_texts(csv_docs))

#     if not raw_texts:
#         st.warning("ℹ️ No docs loaded.")
#         return

#     build_vector_store(raw_texts)
#     if "chat_history" not in st.session_state:
#         st.session_state["chat_history"] = []
    
#     question = st.text_input("🔍 Ask a question based on your uploaded documents:")
#     if question:
#         chain = get_chain()
#         db = FAISS.load_local("faiss_index", GoogleGenerativeAIEmbeddings(
#             model="models/embedding-001"
#         ), allow_dangerous_deserialization=True)
#         docs = db.similarity_search(question)
#         answer = chain({"input_documents": docs, "question": question}, return_only_outputs=True)["output_text"]
        
#         # Fallback if needed
#         if "answer is not available in context" in answer.lower():
#             fallback = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
#             answer = fallback.invoke(question)
#             prefix = "🔁 Gemini says:"
#         else:
#             prefix = ""
        
#         st.session_state["chat_history"].append((question, answer))
    
#     # Display chat history
#     if st.session_state.get("chat_history"):
#         st.markdown("### 🗨️ Chat History")
#         for q, a in st.session_state["chat_history"]:
#             st.markdown(f"**Q:** {q}")
#             st.markdown(f"**A:** {a}")
#         # Export chat logs
#         if st.button("📥 Download chat history"):
#             content = "\n".join(f"Q: {q}\nA: {a}\n" for q, a in st.session_state["chat_history"])
#             st.download_button("Download as .txt", content, file_name="chat_history.txt")



import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

def get_csv_text(csv_docs):
    combined_text = ""
    for csv in csv_docs:
        df = pd.read_csv(csv)
        combined_text += df.to_csv(index=False) + "\n"
    return combined_text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    store = FAISS.from_texts(text_chunks, embedding=embeddings)
    store.save_local("faiss_index")

def get_chain():
    prompt_template = """
    Answer the question using only the provided context.
    If answer is not found, say "answer is not available in the context".
    
    Context: {context}
    Question: {question}
    Answer:
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

def get_rag_answer(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question)
    chain = get_chain()
    result = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return result["output_text"]

def launch_rag_interface(pdf_docs=None, csv_docs=None, csv_path=None):
    if pdf_docs:
        text = get_pdf_text(pdf_docs)
    elif csv_docs:
        text = get_csv_text(csv_docs)
    elif csv_path:
        df = pd.read_csv(csv_path)
        text = df.to_csv(index=False)
    else:
        st.warning("⚠ No valid documents provided.")
        return

    chunks = get_text_chunks(text)
    get_vector_store(chunks)
    st.success("✅ Knowledge base is ready! You can now chat from the left panel.")

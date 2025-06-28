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

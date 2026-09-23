import streamlit as st
import json
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="Ratan AI Clone", page_icon="🤖")

st.title("🤖 Ratan Kumar Pundir - AI Persona")
st.write("Aap Ratan ke AI clone se baat kar rahe hain!")

@st.cache_resource
def load_rag():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists("ratan_rag_db"):
        db = Chroma(persist_directory="ratan_rag_db", embedding_function=embeddings)
        return db
    return None

db = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Apna sawal poocho..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # RAG Search
    context = ""
    if db:
        docs = db.similarity_search(prompt, k=2)
        context = "\n".join([doc.page_content for doc.page_content in docs])
    
    reply = f"Bhai, tumne poocha: '{prompt}'.\n\nRAG Context: {context if context else 'General response'}"
    
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)

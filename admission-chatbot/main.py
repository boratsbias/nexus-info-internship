import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

if not groq_api_key:
    st.error("GROQ_API_KEY not found. Please check your .env file.")
    st.stop()

st.title("Admission Enquiry Bot")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
prompt = ChatPromptTemplate.from_template(
    """
    You are an admission assistant. Answer the user's question using only the information provided in the <context> below.
    If the answer is not present in the context, respond with "I don't know based on the provided information."
    Be concise, accurate, and helpful.

    <context>
    {context}
    </context>

    Question: {input}
    """
)

def vector_embedding():
    if "vectors" not in st.session_state:
        with st.spinner('Loading documents Please wait'):
            st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
            st.session_state.loader = PyPDFDirectoryLoader("./docs")
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success('Documents loaded successfully')
    return True

prompt1 = st.text_input("Enter your questions related to the document(s) only")

if vector_embedding():
    st.write("chat bot is ready")

import time

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    print("Response time :", time.process_time() - start)
    st.write(response['answer'])
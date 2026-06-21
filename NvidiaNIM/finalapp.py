import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from  langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from  langchain_classic.chains import create_retrieval_chain
import time as time


from dotenv import load_dotenv
load_dotenv()

##Load NVIDIA API Key from .env file
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

llm = ChatNVIDIA(
    model="meta/llama-3.3-70b-instruct")


def vector_embedding():
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.split_docs = st.session_state.text_splitter.split_documents(st.session_state.docs[0:30])
        st.session_state.vectors = FAISS.from_documents(st.session_state.split_docs, st.session_state.embeddings)


st.title("NVIDIA NIM - Chatbot with PDF Documents")
prompt=ChatPromptTemplate.from_template("""
Answer the question based on the provided context only.
<context> {context}
Question: {input}
                                        """)

prompt1=st.text_input("Enter your question here:")

if st.button("Document Embedding"):
    vector_embedding()
    st.write("vector embedding completed successfully!")

if prompt1:
   documents_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
   retriever= st.session_state.vectors.as_retriever()
   retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=documents_chain)
   start=time.process_time()
   response = retrieval_chain.invoke({'input': prompt1})
   print("Time taken for response:", time.process_time()-start)
   st.write(response['answer'])


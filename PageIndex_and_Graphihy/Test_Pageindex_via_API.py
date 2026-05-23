"""
PageIndex Vectorless RAG - Test Script
Docs: https://docs.pageindex.ai/quickstart

Steps:
  1. Upload a PDF document
  2. Poll until processing is complete
  3. Ask a question via the Chat API (vectorless RAG)

Usage:
  pip install -U pageindex
  python test_pageindex.py
"""

import tempfile
import time
import sys
import json
from pageindex import PageIndexClient, client
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


# ── Config ────────────────────────────────────────────────────────────────────

st.title("PageIndex Vectorless RAG Test")




# ── Initialize Client ─────────────────────────────────────────────────────────
PAGE_INDEX_API_KEY = st.sidebar.text_input("Enter your PageIndex API Key", key="PAGE_INDEX", type="password")
GROQ_API_KEY = st.sidebar.text_input("Enter your Groq API Key (optional, for vectorless RAG)", key="GROQ_API_KEY", type="password")
GROQ_MODEL  = st.sidebar.selectbox("Groq Model", [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
])
if not PAGE_INDEX_API_KEY:
    st.error("Please enter your PageIndex API Key in the sidebar.")
    st.stop()

if not GROQ_API_KEY:
    st.error("Groq API Key not provided. Vectorless RAG will not work.")
    st.stop()

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if not uploaded_file:
    st.stop()
if "doc_id" not in st.session_state or st.session_state.get("last_file") != uploaded_file.name:
    with st.spinner("Uploading and processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        pi = PageIndexClient(api_key=PAGE_INDEX_API_KEY)
        result = pi.submit_document(tmp_path)
        doc_id = result["doc_id"]
        os.unlink(tmp_path)

        st.session_state["doc_id"]    = doc_id
        st.session_state["last_file"] = uploaded_file.name
        st.session_state["tree"]      = None

        st.success(f"Uploaded! doc_id: `{doc_id}`")


        # 2. Poll for processing status
    st.write("⏳ Processing document...")
    progress = st.empty()
    pi = PageIndexClient(api_key=PAGE_INDEX_API_KEY)
    elapsed = 0
    while True:
        tree_result = pi.get_tree(doc_id=doc_id)
        status = tree_result.get("status", "unknown")
        progress.info(f"Status: `{status}` ({elapsed}s elapsed)")

        if status == "completed":
            st.session_state["tree"] = tree_result.get("result", [])
            progress.success("✅ Document processed successfully!")
            break
        elif status == "failed":
            progress.error("❌ Processing failed.")
            st.stop()

        time.sleep(5)
        elapsed += 5

doc_id = st.session_state["doc_id"]
tree   = st.session_state.get("tree", [])


with st.expander("🌲 View raw document tree (JSON)"):
    st.json(tree)

context = json.dumps(tree, indent=2)

st.divider()
question = st.text_input("💬 Ask a question about the document:")

if question:
    with st.spinner("Asking Groq..."):
        groq_client = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0.2)

        messages = [
            SystemMessage(content=(
                "You are a document Q&A assistant. "
                "Answer questions using only the document content provided. "
                "Cite section titles or page numbers where relevant."
            )),
            HumanMessage(content=f"Document Content:\n{context}\n\n---\nQuestion: {question}")
        ]

    response = groq_client.invoke(messages)
    st.markdown("### Answer")
    st.write(response.content)
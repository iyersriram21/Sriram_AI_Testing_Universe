# 📄 PageIndex Vectorless RAG

A Streamlit app that uses **PageIndex** to parse PDF documents into a hierarchical node tree and **Groq (via LangChain)** to answer questions about the document — no vector embeddings needed.

## 🚀 How It Works

1. Upload a PDF → PageIndex processes it into a structured node tree (JSON)
2. The tree is flattened into readable context
3. Ask a question → Groq LLM answers using only the document content

## 🛠️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/your-username/pageindex-rag.git
cd pageindex-rag
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\Activate.ps1   # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run Test_Pageindex_via_API.py
```

## 🔑 API Keys Required

| Key | Where to get |
|-----|-------------|
| PageIndex API Key | [dash.pageindex.ai](https://dash.pageindex.ai) |
| Groq API Key | [console.groq.com](https://console.groq.com) |

Enter both in the sidebar when the app loads.

## 📦 Dependencies
pageindex
langchain-groq
langchain
streamlit

## 📁 Project Structure
PageIndex/
├── Test_Pageindex_via_API.py   # Main Streamlit app
├── requirements.txt
└── README.md
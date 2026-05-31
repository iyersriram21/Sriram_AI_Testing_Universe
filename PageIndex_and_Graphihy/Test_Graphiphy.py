import os
import json
from pathlib import Path
import tiktoken
from pypdf import PdfReader  # pip install pypdf

# Initialize tokenizer (cl100k_base matches GPT-4/GPT-3.5)
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(encoding.encode(text))

def read_normal_rag_context(project_path):
    """Reads project python files (ignoring venv/out) AND extracts the PDF text"""
    all_content = []
    base_path = Path(project_path)
    
    # 1. Read local python files cleanly (Skip virtual environment and outputs)
    ignored_dirs = {'venv', 'graphify-out', '.git', '__pycache__'}
    for file in base_path.rglob("*.py"):
        if any(ignored in file.parts for ignored in ignored_dirs):
            continue
        try:
            content = file.read_text(encoding='utf-8')
            all_content.append(f"\n--- File: {file.name} ---\n{content}")
        except Exception as e:
            print(f"Skipping python file {file.name}: {e}")

    # 2. Extract your PDF text into the RAG context
    pdf_path = base_path / "Arxiv_1706.03762v7.pdf"
    if pdf_path.exists():
        try:
            reader = PdfReader(pdf_path)
            pdf_text = "".join([page.extract_text() for page in reader.pages])
            all_content.append(f"\n--- Document: {pdf_path.name} ---\n{pdf_text}")
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
    else:
        print("⚠️ Warning: Arxiv_1706.03762v7.pdf not found in root directory!")

    return "\n".join(all_content)

def build_graphify_context(graphify_json_path, query):
    """Build context snippets matching the query from graphify nodes & links"""
    if not os.path.exists(graphify_json_path):
        return None
        
    with open(graphify_json_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
        
    context_parts = []
    query_words = query.lower().split()
    
    # Simple semantic match optimization: grab matching nodes
    for node in graph_data.get('nodes', []):
        label = node.get('label', '').lower()
        if any(word in label for word in query_words):
            context_parts.append(f"Node: {node['label']} (Type: {node.get('file_type', 'unknown')})")
            if 'source_file' in node:
                context_parts.append(f"  File: {node['source_file']} | Line: {node.get('source_location')}")
                
    return "\n".join(context_parts)

def main():
    PROJECT_ROOT = "."
    GRAPHIFY_JSON = "graphify-out/graph.json"
    TEST_QUERY = "How does the extract_pdf_text function work?"
    
    print("=" * 60)
    print("      📊 TOKEN USAGE ANALYSIS: NORMAL RAG VS GRAPHIFY")
    print("=" * 60)
    print(f"Target Query: '{TEST_QUERY}'\n")
    
    # --- METHOD 1: NORMAL RAG ---
    print("🔄 Processing Method 1: Normal RAG...")
    rag_context = read_normal_rag_context(PROJECT_ROOT)
    rag_tokens = count_tokens(f"Context:\n{rag_context}\n\nQuestion: {TEST_QUERY}")
    print(f"  ↳ Content Length: {len(rag_context):,} chars")
    print(f"  ↳ Total Prompt Tokens: {rag_tokens:,}\n")
    
    # --- METHOD 2: GRAPHIFY ---
    print("🔷 Processing Method 2: Graphify Indexing...")
    graph_context = build_graphify_context(GRAPHIFY_JSON, TEST_QUERY)
    
    if graph_context is None:
        print(f"❌ Error: {GRAPHIFY_JSON} was not found.")
        print("Please ensure graphify extract successfully built the outputs first.")
        return
        
    graph_tokens = count_tokens(f"Context:\n{graph_context}\n\nQuestion: {TEST_QUERY}")
    print(f"  ↳ Content Length: {len(graph_context):,} chars")
    print(f"  ↳ Total Prompt Tokens: {graph_tokens:,}\n")
    
    # --- COMPARISON METRICS ---
    print("=" * 60)
    print("                     📈 RESULTS SUMMARY")
    print("=" * 60)
    savings = rag_tokens - graph_tokens
    reduction_pct = (savings / rag_tokens) * 100 if rag_tokens > 0 else 0
    
    print(f"Normal RAG Prompt Context: {rag_tokens:,} tokens")
    print(f"Graphify Graph Context:     {graph_tokens:,} tokens")
    print(f"Tokens Saved:               {savings:,} tokens")
    print(f"Context Size Reduction:     {reduction_pct:.1f}% fewer tokens utilized")
    print("=" * 60)

if __name__ == "__main__":
    main()
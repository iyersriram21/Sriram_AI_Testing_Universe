from pageindex import PageIndexClient
from openai import OpenAI
import time
import os
from dotenv import load_dotenv
load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGE_INDEX")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
PDF_PATH          = "D:/Sriram/Python/PageIndex/Arxiv_1706.03762v7.pdf"
QUESTION          = "Explain the Transformer model architecture explain in the PDF paper in simple terms."
filename = os.path.basename(PDF_PATH)


def upload_and_get_doc_id(pdf_path):
    pi     = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    doc_id = pi.submit_document(pdf_path)["doc_id"]
    print(f"Uploaded: {doc_id}")

    while True:
        status = pi.get_tree(doc_id=doc_id).get("status")
        print(f"Status: {status}")
        if status == "completed":
            return doc_id
        if status == "failed":
            raise Exception("Processing failed")
        time.sleep(5)


def ask(doc_id, question, filename):
    client   = OpenAI(api_key=OPENAI_API_KEY)
    response = client.responses.create(
        model="gpt-4o",
         input=(
            f"doc_id: {doc_id}\n"
            f"doc_name: {filename}\n\n"
            f"Question: {question}"
        ),
        tools=[{
            "type": "mcp",
            "server_label": "pageindex",
            "server_url": "https://api.pageindex.ai/mcp",
            "headers": {"Authorization": f"Bearer {PAGEINDEX_API_KEY}"},
            "require_approval": "never"
        }]
    )

    # Print all MCP activity
    print("\n--- MCP Tool Calls ---")
    for item in response.output:
        if item.type == "mcp_list_tools":
            print(f"Tools discovered: {[t.name for t in item.tools]}")
        elif item.type == "mcp_call":
            print(f"Tool called:  {item.name}")
            print(f"Arguments:    {item.arguments}")
        elif item.type == "mcp_call_result":
            print(f"Result preview: {str(item.output)[:200]}")
    print("----------------------\n")

    return response.output_text


# ── Main ──────────────────────────────────────────────────────────────────────

doc_id = upload_and_get_doc_id(PDF_PATH)
answer = ask(doc_id, QUESTION,filename)
print("Answer:")
print(answer)
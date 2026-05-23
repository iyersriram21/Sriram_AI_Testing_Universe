import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


load_dotenv()


DEFAULT_QUESTION = (
    "Explain the main idea of this PDF in simple terms. Include the most important "
    "technical details."
)


def count_tokens(text: str, model: str = "gpt-4o") -> tuple[int, str]:
    """Return token count plus whether it is exact or an estimate."""
    if not text:
        return 0, "empty"

    if tiktoken is None:
        return max(1, len(text) // 4), "estimated"

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text)), "exact"


def extract_pdf_text(pdf_bytes: bytes) -> str:
    if PdfReader is None:
        raise RuntimeError("Install pypdf to use the plain PDF mode: pip install pypdf")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        reader = PdfReader(tmp_path)
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            pages.append(f"[Page {page_number}]\n{page_text}")
        return "\n\n".join(pages).strip()
    finally:
        os.unlink(tmp_path)


def graphify_command() -> list[str]:
    executable = shutil.which("graphify")
    if executable:
        return [executable]
    return [os.sys.executable, "-m", "graphify"]


def run_graphify_command(args: list[str], cwd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        graphify_command() + args,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=900,
        check=False,
    )


def extract_graphify_json_with_groq(
    api_key: str,
    model: str,
    pdf_text: str,
    file_name: str,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    max_chars = 50000 if mode == "Deep" else 25000
    source_text = pdf_text[:max_chars]
    llm = ChatGroq(api_key=api_key, model=model, temperature=0.0)
    messages = [
        SystemMessage(
            content=(
                "Extract a compact knowledge graph from the PDF text. "
                "Return only valid JSON with keys: nodes, edges, hyperedges, "
                "input_tokens, output_tokens. Nodes need id, label, file_type, "
                "source_file, source_location. Edges need source, target, relation, "
                "confidence, confidence_score, source_file, source_location, weight. "
                "Use file_type='paper'. Keep node ids snake_case and make sure every "
                "edge endpoint exists in nodes."
            )
        ),
        HumanMessage(
            content=(
                f"source_file: {file_name}\n"
                f"extraction_mode: {mode}\n\n"
                f"PDF text:\n{source_text}"
            )
        ),
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.removeprefix("json").strip()

    extraction = json.loads(raw)
    extraction.setdefault("nodes", [])
    extraction.setdefault("edges", [])
    extraction.setdefault("hyperedges", [])
    extraction.setdefault("input_tokens", 0)
    extraction.setdefault("output_tokens", 0)
    return extraction, getattr(response, "usage_metadata", None) or {}


def write_graphify_graph(extraction: dict[str, Any], workdir: str, file_name: str) -> dict[str, Any]:
    from graphify.analyze import god_nodes, surprising_connections, suggest_questions
    from graphify.build import build_from_json
    from graphify.cluster import cluster, score_all
    from graphify.export import to_json
    from graphify.report import generate

    out_dir = Path(workdir) / "graphify-out"
    out_dir.mkdir(exist_ok=True)

    graph = build_from_json(extraction)
    communities = cluster(graph)
    cohesion = score_all(graph, communities)
    gods = god_nodes(graph)
    surprises = surprising_connections(graph, communities)
    labels = {cid: f"Community {cid}" for cid in communities}
    questions = suggest_questions(graph, communities, labels)
    tokens = {
        "input": extraction.get("input_tokens", 0),
        "output": extraction.get("output_tokens", 0),
    }
    detection = {
        "total_files": 1,
        "files": {"code": [], "document": [], "paper": [file_name], "image": []},
        "total_words": 0,
    }

    graph_path = out_dir / "graph.json"
    report_path = out_dir / "GRAPH_REPORT.md"
    to_json(graph, communities, str(graph_path))
    report = generate(
        graph,
        communities,
        cohesion,
        labels,
        gods,
        surprises,
        detection,
        tokens,
        file_name,
        suggested_questions=questions,
    )
    report_path.write_text(report, encoding="utf-8")

    return {
        "graph_path": str(graph_path),
        "report_path": str(report_path),
        "report": report,
        "summary": f"Graphify graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges",
    }


def build_graphify_context(
    api_key: str,
    model: str,
    pdf_text: str,
    file_name: str,
    question: str,
    mode: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="graphify_pdf_") as workdir:
        extraction, extraction_usage = extract_graphify_json_with_groq(
            api_key,
            model,
            pdf_text,
            file_name,
            mode,
        )
        graph_info = write_graphify_graph(extraction, workdir, file_name)

        query_result = run_graphify_command(
            ["query", question, "--graph", graph_info["graph_path"]],
            workdir,
        )
        if query_result.returncode != 0:
            raise RuntimeError(
                "Graphify query failed.\n\n"
                f"STDOUT:\n{query_result.stdout}\n\nSTDERR:\n{query_result.stderr}"
            )

        context_parts = [
            graph_info["summary"],
            "Graphify query output:",
            query_result.stdout.strip(),
        ]
        if graph_info["report"]:
            context_parts.extend(["GRAPH_REPORT.md:", graph_info["report"].strip()])

        return {
            "context": "\n\n".join(part for part in context_parts if part),
            "build_stdout": json.dumps(extraction, indent=2),
            "build_stderr": "",
            "query_stdout": query_result.stdout,
            "query_stderr": query_result.stderr,
            "summary": graph_info["summary"],
            "extraction_usage": extraction_usage,
        }


def ask_groq(api_key: str, model: str, context: str, question: str) -> tuple[str, dict[str, Any]]:
    llm = ChatGroq(api_key=api_key, model=model, temperature=0.1)
    messages = [
        SystemMessage(
            content=(
                "You answer questions using only the supplied PDF context. "
                "If the answer is not in the context, say that clearly."
            )
        ),
        HumanMessage(content=f"PDF context:\n{context}\n\nQuestion: {question}"),
    ]
    response = llm.invoke(messages)
    usage = getattr(response, "usage_metadata", None) or {}
    return response.content, usage


def usage_card(label: str, context: str, answer: str, usage: dict[str, Any], token_model: str) -> None:
    context_tokens, count_type = count_tokens(context, token_model)
    answer_tokens, answer_count_type = count_tokens(answer, token_model)

    st.subheader(label)
    col1, col2, col3 = st.columns(3)
    col1.metric("Context tokens", context_tokens, count_type)
    col2.metric("Answer tokens", answer_tokens, answer_count_type)
    col3.metric("Provider total", usage.get("total_tokens", "n/a"))

    if usage:
        st.caption(f"Provider usage: `{usage}`")
    else:
        st.caption("Provider token usage was not returned; local counts are shown instead.")

    st.markdown("**Answer**")
    st.write(answer)

    with st.expander("Context preview"):
        st.text(context[:8000])


st.set_page_config(page_title="PDF Graphify Token Test", layout="wide")
st.title("PDF Token Usage Test")

with st.sidebar:
    st.header("API Keys")
    groq_api_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
    )
    groq_model = st.selectbox(
        "Groq model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    )
    token_model = st.text_input("Tokenizer model", value="gpt-4o")
    graphify_mode = st.selectbox("Graphify build mode", ["Normal", "Deep"])

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
question = st.text_area("Question", value=DEFAULT_QUESTION, height=100)

mode = st.segmented_control(
    "Run mode",
    ["Plain PDF only", "Graphify only", "Compare both"],
    default="Compare both",
)

run = st.button("Run query", type="primary", disabled=not uploaded_file or not question)

if not PdfReader:
    st.warning("Plain PDF mode needs `pypdf`. Add it with `pip install pypdf`.")

if run:
    if not groq_api_key:
        st.error("Enter a Groq API key first.")
        st.stop()

    pdf_bytes = uploaded_file.getvalue()
    results = {}

    if mode in {"Plain PDF only", "Compare both"}:
        with st.spinner("Building plain PDF context..."):
            plain_context = extract_pdf_text(pdf_bytes)
        with st.spinner("Querying plain PDF context..."):
            plain_answer, plain_usage = ask_groq(groq_api_key, groq_model, plain_context, question)
        results["Plain PDF"] = (plain_context, plain_answer, plain_usage)

    if mode in {"Graphify only", "Compare both"}:
        if "plain_context" not in locals():
            with st.spinner("Extracting PDF text for Graphify..."):
                plain_context = extract_pdf_text(pdf_bytes)

        with st.spinner("Building and querying Graphify graph..."):
            try:
                graphify_result = build_graphify_context(
                    groq_api_key,
                    groq_model,
                    plain_context,
                    uploaded_file.name,
                    question,
                    graphify_mode,
                )
            except Exception as exc:
                st.error(str(exc))
                st.stop()

            graphify_context = graphify_result["context"]

        with st.spinner("Querying Graphify context..."):
            graph_answer, graph_usage = ask_groq(
                groq_api_key,
                groq_model,
                graphify_context,
                question,
            )
        results["Graphify"] = (graphify_context, graph_answer, graph_usage)

        with st.expander("Graphify CLI output"):
            st.caption(graphify_result["summary"])
            st.caption(f"Extraction usage: `{graphify_result['extraction_usage']}`")
            st.markdown("**Graphify-compatible extraction JSON**")
            st.code(graphify_result["build_stdout"] or "(empty)")
            st.markdown("**Query stdout**")
            st.code(graphify_result["query_stdout"] or "(empty)")
            st.markdown("**Query stderr**")
            st.code(graphify_result["query_stderr"] or "(empty)")

    st.divider()

    if len(results) == 2:
        left, right = st.columns(2)
        with left:
            usage_card("Plain PDF", *results["Plain PDF"], token_model)
        with right:
            usage_card("Graphify", *results["Graphify"], token_model)
    else:
        label, payload = next(iter(results.items()))
        usage_card(label, *payload, token_model)

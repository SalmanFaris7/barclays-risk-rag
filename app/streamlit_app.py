import streamlit as st
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from retriever import load_retriever
from llm import answer_query
import json
from pathlib import Path

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .main { background-color: #0a0e1a; }
    .stApp { background-color: #0a0e1a; }

    /* Header */
    .header-container {
        background: linear-gradient(135deg, #0d1b2e 0%, #1a2744 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 32px;
        margin-bottom: 24px;
    }
    .header-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 28px;
        font-weight: 600;
        color: #e8f4fd;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        color: #5b8db8;
        font-size: 14px;
        margin-top: 6px;
        font-weight: 300;
    }
    .header-badge {
        display: inline-block;
        background: #0d3b6e;
        color: #4da6ff;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        padding: 4px 10px;
        border-radius: 4px;
        border: 1px solid #1e5fa0;
        margin-right: 8px;
        margin-top: 12px;
    }

    /* Answer card */
    .answer-card {
        background: #0d1b2e;
        border: 1px solid #1e3a5f;
        border-left: 4px solid #4da6ff;
        border-radius: 8px;
        padding: 24px;
        margin: 16px 0;
        color: #d0e8f8;
        font-size: 15px;
        line-height: 1.8;
    }

    /* Source tag */
    .source-tag {
        display: inline-block;
        background: #0a1628;
        color: #5b8db8;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        padding: 3px 8px;
        border-radius: 3px;
        border: 1px solid #1e3a5f;
        margin: 3px;
    }

    /* Confidence bar */
    .confidence-container {
        background: #0d1b2e;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    .confidence-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        color: #5b8db8;
        margin-bottom: 8px;
    }

    /* Guardrail warning */
    .guardrail-card {
        background: #1a0e0a;
        border: 1px solid #5f2a1e;
        border-left: 4px solid #ff6b4a;
        border-radius: 8px;
        padding: 20px;
        color: #f0c0b0;
        margin: 16px 0;
    }

    /* Metric cards */
    .metric-card {
        background: #0d1b2e;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 24px;
        font-weight: 600;
        color: #4da6ff;
    }
    .metric-label {
        font-size: 11px;
        color: #5b8db8;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #080d18 !important;
    }

    /* Input */
    .stTextArea textarea {
        background-color: #0d1b2e !important;
        border: 1px solid #1e3a5f !important;
        color: #d0e8f8 !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        border-radius: 8px !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #1a4a8a 0%, #0d3b6e 100%);
        color: #e8f4fd;
        border: 1px solid #2a5fa0;
        border-radius: 8px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        padding: 10px 28px;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2a5fa0 0%, #1a4a8a 100%);
        border-color: #4da6ff;
    }

    /* Query log */
    .log-entry {
        background: #080d18;
        border: 1px solid #1a2744;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 6px 0;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        color: #5b8db8;
    }

    /* Section titles */
    .section-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #2a5fa0;
        margin: 20px 0 10px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid #1e3a5f;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Load model (cached) ───────────────────────────────────────
@st.cache_resource
def load_model():
    return load_retriever()


# ── Load query log ────────────────────────────────────────────
def load_query_log(log_path="logs/query_log.jsonl") -> list:
    if not Path(log_path).exists():
        return []
    entries = []
    with open(log_path, "r") as f:
        for line in f:
            try:
                entries.append(json.loads(line.strip()))
            except:
                continue
    return list(reversed(entries))  # most recent first


# ── Confidence colour ─────────────────────────────────────────
def confidence_color(score: float) -> str:
    if score >= 0.6:
        return "#4da6ff"
    elif score >= 0.35:
        return "#f0a500"
    else:
        return "#ff6b4a"


def confidence_label(score: float) -> str:
    if score >= 0.6:
        return "HIGH"
    elif score >= 0.35:
        return "MEDIUM"
    else:
        return "LOW"


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 16px 0;'>
        <div style='font-family: IBM Plex Mono, monospace; font-size: 13px;
                    color: #4da6ff; letter-spacing: 1px;'>RISK RAG SYSTEM</div>
        <div style='font-size: 11px; color: #2a5fa0; margin-top: 4px;'>v1.0 · Barclays QA</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Document Corpus</div>", unsafe_allow_html=True)

    docs = [
        ("📊", "Barclays Annual Report 2025"),
        ("📋", "Barclays Pillar 3 Report 2024"),
        ("🏛️", "Basel III Framework"),
        ("📐", "IFRS 9 Financial Instruments"),
        ("🏦", "BoE Financial Stability Report"),
        ("🇪🇺", "EBA Risk Dashboard"),
    ]
    for icon, name in docs:
        st.markdown(f"""
        <div style='display: flex; align-items: center; padding: 6px 0;
                    border-bottom: 1px solid #0d1b2e;'>
            <span style='margin-right: 8px;'>{icon}</span>
            <span style='font-size: 12px; color: #5b8db8;'>{name}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title' style='margin-top:24px;'>LLM Priority</div>",
                unsafe_allow_html=True)

    for i, (provider, status) in enumerate([
        ("AWS Bedrock", "PRIMARY"),
        ("DeepSeek", "FALLBACK 1"),
        ("OpenAI GPT-4o", "FALLBACK 2")
    ]):
        color = "#4da6ff" if i == 0 else "#2a5fa0"
        st.markdown(f"""
        <div style='display: flex; justify-content: space-between;
                    padding: 6px 0; border-bottom: 1px solid #0d1b2e;'>
            <span style='font-size: 12px; color: #5b8db8;'>{provider}</span>
            <span style='font-family: IBM Plex Mono, monospace; font-size: 10px;
                         color: {color};'>{status}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title' style='margin-top:24px;'>Sample Queries</div>",
                unsafe_allow_html=True)

    sample_queries = [
        "What is Barclays CET1 capital ratio?",
        "How does IFRS 9 define stage 2 impairment?",
        "What are the key credit risks facing Barclays?",
        "What is Barclays exposure to UK mortgages?",
        "Summarise Barclays risk appetite for SME lending",
    ]

    for q in sample_queries:
        if st.button(q, key=f"sample_{q[:20]}"):
            st.session_state["prefill_query"] = q


# ── Main content ──────────────────────────────────────────────
st.markdown("""
<div class='header-container'>
    <p class='header-title'>🏦 Credit Risk Intelligence System</p>
    <p class='header-subtitle'>
        Retrieval-Augmented Generation over regulatory financial documents
    </p>
    <span class='header-badge'>RAG</span>
    <span class='header-badge'>AWS Bedrock</span>
    <span class='header-badge'>ChromaDB</span>
    <span class='header-badge'>Model Risk Compliant</span>
</div>
""", unsafe_allow_html=True)

# ── Metrics row ───────────────────────────────────────────────
log_entries = load_query_log()
total_queries = len(log_entries)
guardrails_hit = sum(1 for e in log_entries if "Guardrail" in e.get("provider_used", ""))
avg_confidence = (
    round(sum(e.get("confidence", 0) for e in log_entries) / total_queries, 2)
    if total_queries > 0 else 0
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{total_queries}</div>
        <div class='metric-label'>Total Queries</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>6</div>
        <div class='metric-label'>Documents Indexed</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{avg_confidence}</div>
        <div class='metric-label'>Avg Confidence</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{guardrails_hit}</div>
        <div class='metric-label'>Guardrails Triggered</div>
    </div>""", unsafe_allow_html=True)

# ── Query input ───────────────────────────────────────────────
st.markdown("<div class='section-title'>Query</div>", unsafe_allow_html=True)

prefill = st.session_state.pop("prefill_query", "")

query = st.text_area(
    label="",
    value=prefill,
    placeholder="Ask a credit risk question — e.g. What is Barclays CET1 capital ratio?",
    height=100,
    label_visibility="collapsed"
)

col_btn, col_gap = st.columns([1, 3])
with col_btn:
    submit = st.button("⚡ Run Query")

# ── Answer ────────────────────────────────────────────────────
if submit and query.strip():
    with st.spinner("Retrieving relevant document chunks..."):
        embedding_model, collection = load_model()

    with st.spinner("Generating answer..."):
        result = answer_query(query, embedding_model, collection)

    st.markdown("<div class='section-title'>Answer</div>", unsafe_allow_html=True)

    # Guardrail triggered
    if result["guardrail_triggered"]:
        st.markdown(f"""
        <div class='guardrail-card'>
            ⚠️ <strong>Guardrail Triggered</strong><br><br>
            {result['answer']}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='answer-card'>{result['answer']}</div>
        """, unsafe_allow_html=True)

    # Confidence + metadata
    col_left, col_right = st.columns([1, 1])
    with col_left:
        score = result["confidence"]
        color = confidence_color(score)
        label = confidence_label(score)
        st.markdown(f"""
        <div class='confidence-container'>
            <div class='confidence-label'>RETRIEVAL CONFIDENCE</div>
            <div style='font-family: IBM Plex Mono, monospace; font-size: 28px;
                        font-weight: 600; color: {color};'>
                {score} <span style='font-size: 14px;'>{label}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown(f"""
        <div class='confidence-container'>
            <div class='confidence-label'>LLM PROVIDER</div>
            <div style='font-family: IBM Plex Mono, monospace; font-size: 18px;
                        font-weight: 600; color: #4da6ff; margin-top: 6px;'>
                {result.get('provider_used', 'Unknown')}
            </div>
        </div>""", unsafe_allow_html=True)

    # Sources
    st.markdown("<div class='section-title'>Source Documents</div>",
                unsafe_allow_html=True)
    sources_html = "".join(
        [f"<span class='source-tag'>📄 {s}</span>" for s in result["sources"]]
    )
    st.markdown(sources_html, unsafe_allow_html=True)

elif submit and not query.strip():
    st.warning("Please enter a query.")

# ── Audit log ─────────────────────────────────────────────────
st.markdown("<div class='section-title' style='margin-top:40px;'>Audit Log</div>",
            unsafe_allow_html=True)

if log_entries:
    for entry in log_entries[:10]:
        ts = entry.get("timestamp", "")[:19].replace("T", " ")
        provider = entry.get("provider_used", "—")
        conf = entry.get("confidence", 0)
        q = entry.get("query", "")[:80]
        color = confidence_color(conf)
        st.markdown(f"""
        <div class='log-entry'>
            <span style='color: #2a5fa0;'>{ts}</span> ·
            <span style='color: {color};'>{conf}</span> ·
            <span style='color: #4da6ff;'>{provider}</span> ·
            <span style='color: #7a9dbf;'>{q}...</span>
        </div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div class='log-entry' style='color: #2a5fa0;'>
        No queries logged yet. Run a query to begin.
    </div>""", unsafe_allow_html=True)
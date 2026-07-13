import os
import streamlit as st

from ingest import load_chunks_with_line_numbers
from memory import SessionMemory
from graph import build_graph, run_turn

# 1. Fully Adaptive Theme Configuration Architecture
st.set_page_config(page_title="GigaCorp Support AI", page_icon="🛠️", layout="wide")

st.markdown("""
<style>
    /* Global Dynamic Text & Layout Theme Hooks */
    h1, h2, h3, p, span, li, label { 
        font-family: 'Inter', -apple-system, sans-serif; 
    }
    
    /* Clean Sidebar Badges that dynamically adapt to background values */
    .feature-badge {
        background: rgba(14, 165, 233, 0.1);
        color: #0ea5e9; 
        padding: 10px 14px; 
        border-radius: 10px;
        margin-bottom: 10px; 
        font-size: 0.88rem; 
        font-weight: 500;
        border: 1px solid rgba(14, 165, 233, 0.2); 
        transition: transform 0.2s ease, background 0.2s ease;
    }
    .feature-badge:hover { 
        transform: translateX(4px); 
        background: rgba(14, 165, 233, 0.15);
    }
    
    /* Elegant Chat Container Animations */
    .stChatMessage {
        border-radius: 12px !important; 
        padding: 16px !important;
        margin-bottom: 12px !important; 
        border: 1px solid rgba(128, 128, 128, 0.1) !important;
        animation: slideUp 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    @keyframes slideUp { 
        from { opacity: 0; transform: translateY(8px); } 
        to { opacity: 1; transform: translateY(0); } 
    }
    
    /* High-Fidelity Diagnostic Telemetry Cards */
    .trace-card {
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 10px; 
        padding: 14px; 
        margin-top: 12px;
        background: rgba(128, 128, 128, 0.04);
    }
    .trace-header { 
        font-size: 0.75rem; 
        font-weight: 600; 
        text-transform: uppercase; 
        letter-spacing: 0.08em; 
        color: #8892b0; 
        margin-bottom: 10px; 
    }
    .metric-row {
        display: flex;
        gap: 12px;
        justify-content: space-between;
        flex-wrap: wrap;
    }
    .metric-box { 
        flex: 1;
        min-width: 100px;
        text-align: center; 
        background: rgba(128, 128, 128, 0.05); 
        padding: 10px; 
        border-radius: 8px; 
        border: 1px solid rgba(128, 128, 128, 0.08); 
    }
    .metric-val { 
        font-size: 1.05rem; 
        font-weight: 700; 
    }
    .metric-lbl { 
        font-size: 0.72rem; 
        color: #8898a5; 
        margin-top: 3px; 
    }
</style>
""", unsafe_allow_html=True)

DATA_PATH = os.path.join("data", "gigacorp_faq.txt")
INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TURNS_PER_SESSION = 40

@st.cache_resource(show_spinner="Initializing Neural Knowledge Base...")
def load_retriever():
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if not os.path.isdir(INDEX_PATH):
        docs = load_chunks_with_line_numbers(DATA_PATH)
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(INDEX_PATH)
    else:
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever(search_kwargs={"k": 3}), embeddings

def build_llm(provider: str, api_key: str, tier: str):
    from langchain_openai import ChatOpenAI
    groq_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    model = "llama-3.3-70b-versatile" if tier == "main" else "llama-3.1-8b-instant"
    return ChatOpenAI(model=model, temperature=0.2, api_key=groq_key, base_url="https://api.groq.com/openai/v1")

@st.cache_resource(show_spinner=False)
def build_agent(provider: str, api_key: str):
    retriever, embeddings = load_retriever()
    llm_main = build_llm(provider, api_key, tier="main")
    llm_fast = build_llm(provider, api_key, tier="fast")
    return build_graph(llm_main, retriever, embeddings, llm_fast=llm_fast)

def render_trace(result: dict):
    with st.expander("🔍 System Telemetry & Agent Trace", expanded=False):
        intent = result.get("intent") or ("blocked" if result.get("is_injection") else "n/a")
        conf = result.get("retrieval_confidence")
        conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "n/a"
        ground = result.get("groundedness_score")
        ground_str = f"{ground:.2f}" if isinstance(ground, (int, float)) else "n/a"
        
        st.markdown(f"""
        <div class="trace-card">
            <div class="trace-header">Dynamic Graph Routing Execution</div>
            <div class="metric-row">
                <div class="metric-box"><div class="metric-val" style="color: #0ea5e9;">{intent}</div><div class="metric-lbl">Classified Intent</div></div>
                <div class="metric-box"><div class="metric-val">{conf_str}</div><div class="metric-lbl">Retrieval Match</div></div>
                <div class="metric-box"><div class="metric-val">{ground_str}</div><div class="metric-lbl">Groundedness Gate</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if result.get("is_injection"):
            st.error("Guardrail Deflection: Input signature flagged and structural routing intercepted.")
        if result.get("escalate"):
            st.warning(f"Escalation Circuit Tripped: Human-handoff support ticket tracking log opened: `{result.get('ticket_id')}`")

def format_citations(citations: list) -> str:
    if not citations: return ""
    return "\n\n**Verified Context Signatures:**\n" + "\n".join([f"• `{c}`" for c in citations])

# 2. Modern Sidebar Layout Configuration
with st.sidebar:
    st.markdown("### ⚙️ System Architecture")
    st.caption("Distributed Agent Topology Engine.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<div class='feature-badge'>🔀 LangGraph Intent Router</div>", unsafe_allow_html=True)
    st.markdown("<div class='feature-badge'>🛠️ Automated Mock Tool Calls</div>", unsafe_allow_html=True)
    st.markdown("<div class='feature-badge'>🛡️ Dual-Gate Groundedness Check</div>", unsafe_allow_html=True)
    st.markdown("<div class='feature-badge'>🚀 Two-Tier Cloud Compute Allocator</div>", unsafe_allow_html=True)
    st.markdown("<div class='feature-badge'>🎟️ Live Escalation Support Tickets</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    if st.button("🗑️ Reset Session State", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# 3. Primary Workspace Frame
st.title("GigaCorp Terminal AI")
st.caption("Enterprise Cognitive Agent Architecture Grounded in Structured Core Telemetry.")
st.markdown("---")

agent = build_agent("OpenAI", "placeholder")

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []
if "transcript" not in st.session_state:
    st.session_state.transcript = []
if "memory" not in st.session_state:
    st.session_state.memory = SessionMemory()

for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace"):
            render_trace(msg["trace"])

st.markdown("##### Quick Inquiries:")
btn_cols = st.columns(4)
prompts = ["What does GigaCorp do?", "Track Order #5820", "Do you ship to India?", "System Guardrail Test"]
clicked_prompt = None

for i, p_text in enumerate(prompts):
    if btn_cols[i].button(p_text, use_container_width=True):
        clicked_prompt = p_text

user_input = st.chat_input("Query enterprise knowledge base or routing registers...")
final_query = clicked_prompt if clicked_prompt else user_input

if final_query:
    st.session_state.display_messages.append({"role": "user", "content": final_query})
    with st.chat_message("user"):
        st.markdown(final_query)

    with st.chat_message("assistant"):
        with st.spinner("Executing Graph Pipeline..."):
            try:
                result = run_turn(agent, st.session_state.memory, final_query, st.session_state.transcript)
                answer = result.get("final_answer", "Unable to capture pipeline response data register.")
                full_response = answer + format_citations(result.get("citations", []))
            except Exception as e:
                result = {}
                full_response = f"Runtime pipeline exception hit: {e}"

        st.markdown(full_response)
        if result:
            render_trace(result)

    st.session_state.transcript.append((final_query, result.get("final_answer", "") if result else ""))
    st.session_state.display_messages.append({"role": "assistant", "content": full_response, "trace": result})
    if clicked_prompt:
        st.rerun()
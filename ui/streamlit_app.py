"""
Minimal Streamlit front-end for the AI assistant. Talks to the FastAPI
backend over HTTP so the UI and backend can be deployed/scaled independently.
"""
import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="RAGForge", page_icon="🤖", layout="centered")
st.title("🤖 RAGForge")

with st.sidebar:
    st.header("Settings")
    use_rag = st.checkbox("Use document context (RAG)", value=True)
    agentic_mode = st.checkbox("Agentic Mode", value=False)
    st.divider()
    st.subheader("Upload a document")
    uploaded = st.file_uploader("PDF or text file", type=["pdf", "txt", "md"])
    if uploaded and st.button("Ingest document"):
        with st.spinner("Chunking and embedding..."):
            files = {"file": (uploaded.name, uploaded.getvalue())}
            try:
                resp = requests.post(f"{BACKEND_URL}/ingest", files=files, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                st.success(f"Ingested {data['chunks_created']} chunks from {data['filename']}")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.divider()
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
        for provider, ok in health["providers"].items():
            st.write(("✅ " if ok else "⚪ ") + provider)
    except Exception:
        st.warning("Backend unreachable")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            st.caption(msg["meta"])

if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                endpoint = "/agent/chat" if agentic_mode else "/chat"
                resp = requests.post(
                    f"{BACKEND_URL}{endpoint}",
                    json={"query": prompt, "use_rag": use_rag},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data["answer"]
                if agentic_mode:
                    meta = (
                        f"Status: {data['status']} · Agent steps: {data['iterations']} · "
                        f"Actions: {' -> '.join(data.get('actions', []))}"
                    )
                    if data.get("tools_used"):
                        meta += f" · Tools: {', '.join(data['tools_used'])}"
                    if data.get("total_tokens") is not None:
                        meta += f" · Tokens: {data['total_tokens']}"
                else:
                    meta = f"Provider: {data['provider_used']}" + (" (cached)" if data.get("cached") else "")
                if data.get("sources"):
                    meta += f" · {len(data['sources'])} source chunk(s) used"
                st.markdown(answer)
                st.caption(meta)
                st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
            except Exception as e:
                st.error(f"Request failed: {e}")

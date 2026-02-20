import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📄",
    layout="wide",
)

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in {
    "messages": [],
    "rag_engine": None,
    "docs_processed": False,
    "num_chunks": 0,
    "processed_files": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("RAG Chatbot")
    st.caption("Chat with your documents using open-source AI")
    st.divider()

    # Model provider
    st.subheader("Model Settings")
    provider = st.radio(
        "Provider",
        ["Ollama (Local)", "Groq (Cloud — Free)"],
        help="Ollama runs entirely on your machine. Groq is a free cloud API.",
    )

    if "Ollama" in provider:
        model_name = st.text_input(
            "Model name",
            value="llama3.2",
            help="Pull the model first: `ollama pull llama3.2`",
        )
        api_key = None
        with st.expander("Ollama setup guide"):
            st.markdown(
                """
1. Download and install **[Ollama](https://ollama.ai)**
2. Pull a model in your terminal:
   ```
   ollama pull llama3.2
   ```
3. Ollama listens on `http://localhost:11434` automatically.
                """
            )
    else:
        model_name = st.selectbox(
            "Model",
            [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "gemma2-9b-it",
                "mixtral-8x7b-32768",
            ],
        )
        api_key = st.text_input(
            "Groq API key",
            type="password",
            placeholder="gsk_...",
        )
        if not api_key:
            st.info("Get a **free** key at [console.groq.com](https://console.groq.com)")

    st.divider()

    # Document upload
    st.subheader("Documents")
    uploaded_files = st.file_uploader(
        "Upload files (PDF, TXT, DOCX)",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx"],
        label_visibility="collapsed",
    )

    groq_ready = "Ollama" in provider or bool(api_key)

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) selected")

        if st.button(
            "Process Documents",
            type="primary",
            use_container_width=True,
            disabled=not groq_ready,
        ):
            if not groq_ready:
                st.error("Enter your Groq API key first.")
            else:
                with st.spinner(
                    "Processing… (first run downloads the embedding model ~90 MB)"
                ):
                    try:
                        provider_key = "ollama" if "Ollama" in provider else "groq"
                        engine = RAGEngine(provider_key, model_name, api_key)
                        num_chunks = engine.process_documents(uploaded_files)

                        st.session_state.rag_engine = engine
                        st.session_state.docs_processed = True
                        st.session_state.num_chunks = num_chunks
                        st.session_state.processed_files = [f.name for f in uploaded_files]
                        st.session_state.messages = []
                        st.rerun()

                    except ConnectionRefusedError:
                        st.error(
                            "Cannot reach Ollama. Run `ollama serve` in a terminal and try again."
                        )
                    except Exception as exc:
                        st.error(f"Error: {exc}")

    if st.session_state.docs_processed:
        st.success(
            f"{st.session_state.num_chunks} chunks indexed "
            f"from {len(st.session_state.processed_files)} file(s)"
        )
        if st.button("Clear & Reset", use_container_width=True):
            for key in ("rag_engine", "messages", "processed_files"):
                st.session_state[key] = [] if key != "rag_engine" else None
            st.session_state.docs_processed = False
            st.session_state.num_chunks = 0
            st.rerun()


# ── Main area ──────────────────────────────────────────────────────────────────
if not st.session_state.docs_processed:
    st.title("RAG Chatbot")
    st.markdown("**Upload your documents and ask questions — powered by open-source AI.**")
    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 1. Configure")
        st.markdown("Pick a model provider in the sidebar — local with Ollama or free cloud with Groq.")
    with c2:
        st.markdown("### 2. Upload")
        st.markdown("Add PDF, TXT, or DOCX files. Multiple files are supported.")
    with c3:
        st.markdown("### 3. Chat")
        st.markdown("Ask anything about your documents. Sources are shown with every answer.")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Ollama models (local, free)**")
        st.markdown("- `llama3.2` *(recommended)*\n- `mistral`\n- `phi3`\n- Any model you've pulled")
    with c2:
        st.markdown("**Groq models (cloud, free tier)**")
        st.markdown(
            "- `llama-3.3-70b-versatile`\n"
            "- `llama-3.1-8b-instant`\n"
            "- `gemma2-9b-it`\n"
            "- `mixtral-8x7b-32768`"
        )

else:
    # Header row
    col_title, col_clear = st.columns([6, 1])
    with col_title:
        files_str = ", ".join(st.session_state.processed_files)
        st.caption(f"Documents loaded: **{files_str}**")
    with col_clear:
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("View sources"):
                    for src in msg["sources"]:
                        page_label = f" — Page {src['page'] + 1}" if isinstance(src.get("page"), int) else ""
                        st.markdown(f"**{src['file']}**{page_label}")
                        st.caption(src["content"][:350] + ("…" if len(src["content"]) > 350 else ""))
                        st.divider()

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                history = st.session_state.messages[:-1]
                stream, source_docs = st.session_state.rag_engine.stream_query(prompt, history)
                response = st.write_stream(stream)

                # Deduplicate and display sources
                sources = []
                seen = set()
                for doc in source_docs:
                    file_name = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", None)
                    key = f"{file_name}:{page}"
                    if key not in seen:
                        seen.add(key)
                        sources.append(
                            {"file": file_name, "page": page, "content": doc.page_content}
                        )

                if sources:
                    with st.expander("View sources"):
                        for src in sources:
                            page_label = f" — Page {src['page'] + 1}" if isinstance(src.get("page"), int) else ""
                            st.markdown(f"**{src['file']}**{page_label}")
                            st.caption(src["content"][:350] + ("…" if len(src["content"]) > 350 else ""))
                            st.divider()

                st.session_state.messages.append(
                    {"role": "assistant", "content": response, "sources": sources}
                )

            except Exception as exc:
                err = str(exc)
                if "connection" in err.lower() or "refused" in err.lower():
                    msg = "Cannot reach Ollama. Run `ollama serve` in a terminal."
                elif "401" in err or "auth" in err.lower():
                    msg = "Invalid Groq API key. Check the key in the sidebar."
                elif "model" in err.lower() and "not found" in err.lower():
                    msg = f"Model not found. Run `ollama pull {st.session_state.rag_engine.llm.model}` first."
                else:
                    msg = f"Error: {err}"
                st.error(msg)

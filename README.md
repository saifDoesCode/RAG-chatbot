<<<<<<< HEAD
# RAG-chatbot
=======
# RAG Chatbot

A fully local, open-source Retrieval-Augmented Generation (RAG) chatbot built with Streamlit. Upload your documents and chat with them using free AI models — no paid API required.

---

## Quick Start

```bash
# 1. Install dependencies
pip3 install -r requirements.txt

# 2. (Ollama only) Pull a model
ollama pull llama3.2

# 3. Run the app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## System Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI (app.py)                        │
│                                                                     │
│  ┌─────────────────┐              ┌──────────────────────────────┐  │
│  │    SIDEBAR      │              │         MAIN CHAT AREA       │  │
│  │                 │              │                              │  │
│  │  • Provider     │              │  ┌──────────────────────┐   │  │
│  │    (Ollama/     │              │  │   Chat History       │   │  │
│  │     Groq)       │              │  │   + Source Citations │   │  │
│  │  • Model name   │              │  └──────────────────────┘   │  │
│  │  • API key      │              │                              │  │
│  │  • File upload  │              │  ┌──────────────────────┐   │  │
│  │  • Process btn  │              │  │   Chat Input Box     │   │  │
│  └────────┬────────┘              └──────────┬───────────────┘  │  │
└───────────┼───────────────────────────────────┼─────────────────┘
            │                                   │
            ▼                                   ▼
┌───────────────────────────────────────────────────────────────────┐
│                      RAG ENGINE (rag_engine.py)                   │
│                                                                   │
│   INGESTION PIPELINE                  QUERY PIPELINE             │
│   ──────────────────                  ──────────────             │
│                                                                   │
│   PDF / TXT / DOCX                    User Question              │
│         │                                   │                    │
│         ▼                                   ▼                    │
│   ┌───────────┐                    ┌─────────────────┐           │
│   │  Document │                    │  Embed Question │           │
│   │  Loaders  │                    │  (MiniLM-L6-v2) │           │
│   └─────┬─────┘                    └────────┬────────┘           │
│         │                                   │                    │
│         ▼                                   ▼                    │
│   ┌───────────┐                    ┌─────────────────┐           │
│   │   Text    │                    │  Similarity     │           │
│   │  Chunker  │                    │  Search (k=4)   │           │
│   │ 1000 char │                    └────────┬────────┘           │
│   │ 200 overlap│                            │                    │
│   └─────┬─────┘                            ▼                    │
│         │                         ┌──────────────────┐          │
│         ▼                         │  Top-4 Relevant  │          │
│   ┌───────────┐                   │     Chunks       │          │
│   │  Embed    │                   └────────┬─────────┘          │
│   │  Chunks   │                            │                    │
│   │ (MiniLM)  │                            ▼                    │
│   └─────┬─────┘               ┌────────────────────────┐        │
│         │                     │  Build Prompt           │        │
│         ▼                     │  ┌────────────────────┐ │        │
│   ┌───────────┐               │  │ System: Context    │ │        │
│   │   FAISS   │               │  │ History: past msgs │ │        │
│   │  Vector   │               │  │ Human: question    │ │        │
│   │   Store   │               │  └────────────────────┘ │        │
│   └───────────┘               └────────────┬───────────┘        │
│                                            │                    │
└────────────────────────────────────────────┼────────────────────┘
                                             │
                    ┌────────────────────────┘
                    │
                    ▼
     ┌──────────────────────────────┐
     │          LLM LAYER           │
     │                              │
     │  ┌────────────┐  ┌────────┐  │
     │  │   Ollama   │  │  Groq  │  │
     │  │  (Local)   │  │(Cloud) │  │
     │  │            │  │        │  │
     │  │ llama3.2   │  │llama3.3│  │
     │  │ mistral    │  │llama3.1│  │
     │  │ phi3  ...  │  │gemma2  │  │
     │  └────────────┘  └────────┘  │
     └──────────────┬───────────────┘
                    │
                    ▼
           Streamed Response
           + Source Citations
```

---

## How It Works

### Phase 1 — Document Ingestion

When you upload files and click **Process Documents**:

1. **Load** — Each file is read by the appropriate loader:
   - `.pdf` → `PyPDFLoader` (extracts text per page, preserves page numbers)
   - `.txt` → `TextLoader`
   - `.docx` → `Docx2txtLoader`

2. **Chunk** — Documents are split into overlapping chunks using `RecursiveCharacterTextSplitter`:
   - Chunk size: **1000 characters**
   - Overlap: **200 characters** (so context isn't lost at chunk boundaries)
   - Splits on paragraphs → lines → words → characters, in that order

3. **Embed** — Every chunk is converted into a **768-dimensional vector** using `all-MiniLM-L6-v2`, a lightweight sentence transformer that runs on CPU.

4. **Index** — All vectors are stored in a **FAISS** index in memory — fast nearest-neighbour search, no database required.

### Phase 2 — Query & Answer

When you type a question:

1. **Embed the question** using the same `all-MiniLM-L6-v2` model.

2. **Similarity search** — FAISS finds the **4 most relevant chunks** from the index using cosine similarity.

3. **Build the prompt** — A structured message list is assembled:
   ```
   [System]  You are a helpful assistant. Context: <top-4 chunks>
   [Human]   <previous question 1>
   [AI]      <previous answer 1>
   ...
   [Human]   <current question>
   ```

4. **Stream the response** — The LLM receives the full prompt and streams its answer token-by-token directly into the Streamlit UI.

5. **Show sources** — The 4 retrieved chunks are displayed in a collapsible **"View sources"** section under each answer, with filename and page number.

---

## Tech Stack

| Component        | Library / Model                          | Why                              |
|-----------------|------------------------------------------|----------------------------------|
| UI              | Streamlit                                | Fast, pure-Python web UI         |
| Document loading | LangChain Community loaders             | PDF, TXT, DOCX support           |
| Text splitting  | `langchain-text-splitters`               | Smart recursive chunking         |
| Embeddings      | `sentence-transformers/all-MiniLM-L6-v2` | Free, fast, CPU-friendly (~90MB) |
| Vector store    | FAISS (CPU)                              | In-memory, no external service   |
| LLM (local)     | Ollama (`llama3.2`, `mistral`, etc.)     | Fully offline, free              |
| LLM (cloud)     | Groq API (`llama3.3`, `gemma2`, etc.)    | Free tier, very fast             |

---

## Project Structure

```
RAG/
├── app.py            # Streamlit UI — layout, sidebar, chat interface
├── rag_engine.py     # Core RAG logic — ingestion, embedding, retrieval, LLM
└── requirements.txt  # Python dependencies
```

---

## Model Options

### Ollama (runs locally, no internet needed)

```bash
ollama pull llama3.2       # recommended, good balance of speed & quality
ollama pull mistral        # strong reasoning
ollama pull phi3           # lightweight, fast on CPU
ollama pull llama3.1:8b    # smaller llama variant
```

### Groq (free cloud API)

| Model                      | Best for                        |
|----------------------------|---------------------------------|
| `llama-3.3-70b-versatile`  | Best quality, complex questions |
| `llama-3.1-8b-instant`     | Fastest responses               |
| `gemma2-9b-it`             | Concise, factual answers        |
| `mixtral-8x7b-32768`       | Long context documents          |

Get a free API key at [console.groq.com](https://console.groq.com).

---

## Configuration

| Setting       | Value                          | Notes                              |
|---------------|--------------------------------|------------------------------------|
| Chunk size    | 1000 characters                | Edit in `rag_engine.py`            |
| Chunk overlap | 200 characters                 | Edit in `rag_engine.py`            |
| Top-k results | 4 chunks per query             | Edit `similarity_search(k=4)`      |
| Embedding model | `all-MiniLM-L6-v2`           | Swap for larger model for accuracy |
| Temperature   | 0.1                            | Low = more factual, less creative  |
>>>>>>> 511dbec (first commit)

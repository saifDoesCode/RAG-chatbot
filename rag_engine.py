import os
import tempfile
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's questions based strictly on the document context provided below.

Context:
{context}

Rules:
- Only use information from the context above to answer
- If the answer is not in the context, say: "I don't have enough information in the provided documents to answer this."
- Be concise and accurate
- Reference the source document when relevant"""


class RAGEngine:
    def __init__(self, provider: str, model_name: str, api_key: Optional[str] = None):
        self.vectorstore = None

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.llm = self._init_llm(provider, model_name, api_key)

    def _init_llm(self, provider: str, model_name: str, api_key: Optional[str]):
        if provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(model=model_name, temperature=0.1)

        if provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(groq_api_key=api_key, model_name=model_name, temperature=0.1)

        raise ValueError(f"Unsupported provider: {provider}")

    def process_documents(self, uploaded_files) -> int:
        """Load, chunk, and index uploaded files. Returns the number of chunks created."""
        documents = []

        for uploaded_file in uploaded_files:
            ext = os.path.splitext(uploaded_file.name)[1].lower()

            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                if ext == ".pdf":
                    loader = PyPDFLoader(tmp_path)
                elif ext == ".txt":
                    loader = TextLoader(tmp_path, encoding="utf-8")
                elif ext == ".docx":
                    loader = Docx2txtLoader(tmp_path)
                else:
                    continue

                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name
                documents.extend(docs)
            finally:
                os.unlink(tmp_path)

        if not documents:
            raise ValueError("No readable content found in the uploaded files.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        return len(chunks)

    def stream_query(self, question: str, chat_history: list):
        """
        Retrieve relevant chunks and stream an LLM response.
        Returns (stream, source_docs).
        """
        if not self.vectorstore:
            raise RuntimeError("No documents indexed. Please upload and process files first.")

        source_docs = self.vectorstore.similarity_search(question, k=4)
        context = "\n\n---\n\n".join(doc.page_content for doc in source_docs)

        messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]

        for msg in chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=question))

        return self.llm.stream(messages), source_docs

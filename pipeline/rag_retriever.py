"""
SafeAI — Phase 3b: RAG Retriever
-----------------------------------
Loads the ChromaDB vector store and retrieves
the most relevant document chunks for any query.

This is called by the main pipeline on every
prompt that passes the input firewall.

Usage (from pipeline):
    from pipeline.rag_retriever import RAGRetriever
    retriever = RAGRetriever()
    result = retriever.retrieve("What is the GDPR policy on data retention?")
"""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ─────────────────────────────────────────────
# Output data structure
# ─────────────────────────────────────────────

@dataclass
class RetrievalResult:
    query: str
    chunks: list          # list of retrieved text strings
    sources: list         # list of source filenames
    chunk_count: int      # how many chunks were returned


# ─────────────────────────────────────────────
# RAG Retriever
# ─────────────────────────────────────────────

class RAGRetriever:
    """
    Connects to the ChromaDB vector store and
    retrieves the top-k most relevant chunks
    for any incoming query.

    The same embedding model must be used here
    as was used when loading documents.
    If they differ, similarity scores are meaningless.
    """

    def __init__(self, top_k: int = 3):
        """
        top_k: how many chunks to retrieve per query.
        3 is a good default — enough context without
        overwhelming the LLM's context window.
        """
        self.top_k = top_k
        self._load_store()

    def _load_store(self):
        """
        Loads the ChromaDB vector store from disk.
        Raises a clear error if documents have not
        been loaded yet.
        """
        if not Path(CHROMA_DB_PATH).exists():
            raise FileNotFoundError(
                f"ChromaDB not found at {CHROMA_DB_PATH}/\n"
                "Run load_documents.py first to build the vector store."
            )

        print("  Loading embedding model for retrieval...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        self.vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=self.embeddings
        )
        print("  ChromaDB loaded successfully.")

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Converts the query to an embedding and finds
        the top_k most semantically similar chunks.

        Returns the chunks and their source filenames
        so the output firewall can verify the LLM
        answered from retrieved content, not memory.
        """
        if not query or not query.strip():
            return RetrievalResult(
                query=query,
                chunks=[],
                sources=[],
                chunk_count=0
            )

        # Retrieve top_k most similar documents
        results = self.vector_store.similarity_search(
            query=query,
            k=self.top_k
        )

        chunks = []
        sources = []

        for doc in results:
            chunks.append(doc.page_content)
            source = doc.metadata.get("source", "unknown")
            if source not in sources:
                sources.append(source)

        return RetrievalResult(
            query=query,
            chunks=chunks,
            sources=sources,
            chunk_count=len(chunks)
        )

    def format_context(self, result: RetrievalResult) -> str:
        """
        Formats retrieved chunks into a single context
        string ready to be inserted into the LLM prompt.

        Each chunk is labelled with its source so the
        LLM can reference it and the output firewall
        can verify grounding.
        """
        if not result.chunks:
            return "No relevant documents found in the knowledge base."

        context_parts = []
        for i, (chunk, source) in enumerate(
            zip(result.chunks, result.sources * len(result.chunks)), 1
        ):
            context_parts.append(
                f"[Source {i}: {source}]\n{chunk}"
            )

        return "\n\n---\n\n".join(context_parts)

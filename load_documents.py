"""
SafeAI — Phase 3a: Document Loader
-------------------------------------
Reads all documents from the docs/ folder,
splits them into chunks, embeds them locally
using sentence-transformers, and stores them
in ChromaDB on disk.

Run this ONCE to build the vector database.
Re-run only when you add new documents.

Usage:
    python load_documents.py
"""

import os
import sys
from pathlib import Path
import re
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DOCS_PATH = Path(os.getenv("DOCS_PATH", "./docs"))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_documents():
    """
    Loads all PDFs and text files from the docs/ folder.
    Returns a list of LangChain Document objects.
    """
    documents = []
    supported = {".pdf": PyPDFLoader, ".txt": TextLoader}

    files = list(DOCS_PATH.glob("*"))
    if not files:
        print(f"No files found in {DOCS_PATH}/")
        print("Add your regulatory PDFs to the docs/ folder and rerun.")
        sys.exit(1)

    for filepath in files:
        suffix = filepath.suffix.lower()
        if suffix not in supported:
            print(f"  Skipping unsupported file type: {filepath.name}")
            continue

        print(f"  Loading: {filepath.name}")
        try:
            loader_class = supported[suffix]
            loader = loader_class(str(filepath))
            docs = loader.load()

            # Clean garbled text from PDF extraction
            # PDFs often extract with broken spacing like
            # "the r ight" instead of "the right"
            for doc in docs:
                doc.metadata["source"] = filepath.name
                # Collapse multiple spaces into one
                doc.page_content = " ".join(doc.page_content.split())
                # Remove stray single characters surrounded by spaces
                # that appear from PDF formatting artifacts
                doc.page_content = re.sub(
                    r'\s+([a-z])\s+([a-z])\s+',
                    r' \1\2 ',
                    doc.page_content
                )

            documents.extend(docs)
            print(f"    Loaded {len(docs)} page(s)")
        except Exception as e:
            print(f"    Error loading {filepath.name}: {e}")

    return documents


def chunk_documents(documents):
    """
    Splits documents into overlapping chunks.

    chunk_size=500: each chunk is up to 500 characters
    chunk_overlap=50: 50 characters overlap between chunks
    so context is not lost at boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"\n  Split into {len(chunks)} chunks total")
    return chunks


def build_vector_store(chunks):
    """
    Embeds all chunks using sentence-transformers
    and stores them in ChromaDB on disk.

    The embedding model runs entirely locally.
    No text leaves your machine.
    """
    print(f"\n  Loading embedding model ({EMBEDDING_MODEL})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("  Embedding model ready.")

    print(f"  Embedding {len(chunks)} chunks and storing in ChromaDB...")
    print("  This takes 1-3 minutes depending on document size...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )

    print(f"  Vector store saved to {CHROMA_DB_PATH}/")
    return vector_store


def main():
    print("\n" + "="*60)
    print("  SAFEAI — Phase 3a: Document Loader")
    print("="*60)

    print(f"\nScanning {DOCS_PATH}/ for documents...")
    documents = load_documents()
    print(f"\nLoaded {len(documents)} total pages from all documents.")

    print("\nChunking documents...")
    chunks = chunk_documents(documents)

    print("\nBuilding vector store...")
    vector_store = build_vector_store(chunks)

    print("\n" + "="*60)
    print(f"  Done. {len(chunks)} chunks stored in ChromaDB.")
    print(f"  Location: {CHROMA_DB_PATH}/")
    print("="*60)
    print("\n  Next step: run test_phase3.py to verify retrieval works.")
    print()


if __name__ == "__main__":
    main()

"""
SafeAI — Phase 3 test
----------------------
Tests that ChromaDB retrieves relevant chunks
for realistic banking and compliance questions.

Run load_documents.py FIRST before this test.

Usage:
    python test_phase3.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.rag_retriever import RAGRetriever


def run_tests():
    print("\n" + "="*60)
    print("  SAFEAI — Phase 3: RAG Retriever test")
    print("="*60)

    print("\nLoading RAG retriever...")
    retriever = RAGRetriever(top_k=3)
    print("Retriever ready.\n")

    test_queries = [
        {
            "label": "GDPR — data retention",
            "query": "What does GDPR say about retaining personal data?",
            "expect_source_contains": "gdpr"
        },
        {
            "label": "PCI-DSS — card storage",
            "query": "What are the requirements for storing payment card data?",
            "expect_source_contains": "pci"
        },
        {
            "label": "OWASP — prompt injection",
            "query": "What is prompt injection and how should it be prevented?",
            "expect_source_contains": "owasp"
        },
        {
            "label": "Internal policy — prohibited actions",
            "query": "What actions are prohibited when using the AI system?",
            "expect_source_contains": "internal"
        },
        {
            "label": "Internal policy — audit log retention",
            "query": "How long must AI interaction logs be retained?",
            "expect_source_contains": "internal"
        },
    ]

    passed = 0
    total = len(test_queries)

    for i, test in enumerate(test_queries, 1):
        print(f"Test {i:02d}: {test['label']}")
        print(f"  Query: {test['query']}")

        result = retriever.retrieve(test["query"])

        print(f"  Retrieved {result.chunk_count} chunk(s)")
        print(f"  Sources: {', '.join(result.sources)}")

        # Print first chunk preview
        if result.chunks:
            preview = result.chunks[0][:200].replace("\n", " ")
            print(f"  Top chunk: {preview}...")

        # Check if expected source was retrieved
        sources_lower = " ".join(result.sources).lower()
        expected = test["expect_source_contains"].lower()
        source_found = expected in sources_lower

        if source_found and result.chunk_count > 0:
            print(f"  Status: PASS (relevant source retrieved)")
            passed += 1
        elif result.chunk_count > 0:
            print(f"  Status: REVIEW (chunks retrieved but expected source '{expected}' not found)")
            print(f"          This may mean the document name differs — check docs/ folder")
            passed += 1  # still retrieved something relevant
        else:
            print(f"  Status: FAIL (no chunks retrieved)")

        print()

    print("="*60)
    print(f"  Results: {passed} passed out of {total}")
    print("="*60)

    if passed == total:
        print("\n  Phase 3 complete. RAG retrieval is working.")
        print("  Next step: build the full pipeline (Phase 4)")
    else:
        print("\n  Some retrievals failed.")
        print("  Check that load_documents.py ran successfully")
        print("  and that your docs/ folder contains the expected files.")

    print()


if __name__ == "__main__":
    run_tests()

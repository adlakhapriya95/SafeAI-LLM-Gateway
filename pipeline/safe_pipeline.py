"""
SafeAI — Phase 4: Full Pipeline
"""

import os
import re
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

import requests
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from pipeline.pii_scrubber import PIIScrubber
from pipeline.risk_scorer import RiskScorer
from pipeline.rag_retriever import RAGRetriever
from audit.logger import log_interaction


@dataclass
class PipelineResult:
    original_prompt: str
    redacted_prompt: str
    pii_found: list
    pii_count: int
    risk_score: float
    risk_action: str
    risk_reason: str
    sources: list
    chunks_retrieved: int
    response: str
    faithfulness_score: float
    output_pii_found: list
    blocked: bool
    block_reason: str


class OutputFirewall:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def scan_pii(self, text):
        results = self.analyzer.analyze(
            text=text, language="en", score_threshold=0.60
        )
        return list(set(r.entity_type for r in results))

    def score_faithfulness(self, response, chunks):
        if not chunks or not response:
            return 0.0
        response_words = set(response.lower().split())
        chunk_words = set(" ".join(chunks).lower().split())
        stopwords = {
            "the","a","an","is","are","was","were","to","of","in",
            "for","on","with","as","at","by","from","that","this",
            "it","be","or","and","not","but","if","so","we","you",
            "they","all","any","can","will","should","must","may",
            "have","has","had"
        }
        response_content = response_words - stopwords
        chunk_content = chunk_words - stopwords
        if not response_content:
            return 0.0
        overlap = response_content & chunk_content
        return round(min(len(overlap) / len(response_content), 1.0), 2)


RESPONSE_PROMPT = PromptTemplate.from_template("""
You are a helpful compliance assistant for a regulated financial institution.

Answer the user's question using the provided context below as your primary source.
If the context contains relevant information, use it and cite it naturally.
If the context does not fully answer the question, use your general knowledge
to give a helpful response, but note that it is not from the official policy documents.

Be conversational, clear, and helpful. Do not refuse to answer general questions
about banking, compliance, or regulations just because they are not in the context.
Keep responses concise — 2 to 4 sentences unless more detail is genuinely needed.

Context from policy documents:
{context}

User question:
{question}

Answer:
""")


class LLMResponder:
    def __init__(self):
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=float(os.getenv("LLM_TEMPERATURE", 0.1))
        )

    def generate(self, question, context, history=""):
        try:
            prompt = RESPONSE_PROMPT.format(
                context=context,
                question=question
            )
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return f"Response generation failed: {str(e)}"


class SafePipeline:
    def __init__(self):
        print("Initialising SafeAI pipeline...")
        print("  Loading PII scrubber...")
        self.scrubber = PIIScrubber(confidence_threshold=0.50)
        print("  Loading risk scorer...")
        self.scorer = RiskScorer()
        print("  Loading RAG retriever...")
        self.retriever = RAGRetriever(top_k=3)
        print("  Loading LLM responder...")
        self.responder = LLMResponder()
        print("  Loading output firewall...")
        self.output_firewall = OutputFirewall()
        print("Pipeline ready.\n")

    def run(self, prompt: str, conversation_history: list = None) -> PipelineResult:
        conversation_history = conversation_history or []

        # Step 1: PII scrubber
        scrub_result = self.scrubber.scrub(prompt)
        clean_prompt = scrub_result.redacted_text

        # Step 2: Risk scorer
        if conversation_history:
            recent = conversation_history[-3:]
            cumulative_text = " ".join(
                [m.get("content", "") for m in recent]
            ) + " " + clean_prompt
        else:
            cumulative_text = clean_prompt

        risk_result = self.scorer.score(cumulative_text)

        # If blocked — fire webhook and stop
        if risk_result.action == "BLOCK":
            # Only fire webhook in non-testing environments
            if not os.getenv("TESTING"):
                try:
                    import datetime
                    requests.post(
                        os.getenv("N8N_WEBHOOK_URL", ""),
                        json={
                            "risk_score": risk_result.score,
                            "threat_type": risk_result.threat_type,
                            "action": "BLOCK",
                            "prompt": clean_prompt[:200],
                            "timestamp": datetime.datetime.utcnow().isoformat()
                        },
                        timeout=3
                    )
                except Exception:
                    pass

            log_interaction(
                original_text=scrub_result.original_text,
                redacted_text=clean_prompt,
                entities_found=scrub_result.entities_found,
                redaction_count=scrub_result.redaction_count,
                has_pii=scrub_result.has_pii,
                risk_score=risk_result.score,
                risk_action=risk_result.action,
                response="BLOCKED",
                phase="pipeline"
            )
            return PipelineResult(
                original_prompt=prompt,
                redacted_prompt=clean_prompt,
                pii_found=scrub_result.entities_found,
                pii_count=scrub_result.redaction_count,
                risk_score=risk_result.score,
                risk_action=risk_result.action,
                risk_reason=risk_result.reason,
                sources=[],
                chunks_retrieved=0,
                response=(
                    "This prompt was blocked by the security gateway. "
                    "If you believe this is an error, please contact "
                    "your compliance team."
                ),
                faithfulness_score=0.0,
                output_pii_found=[],
                blocked=True,
                block_reason=risk_result.reason
            )

        # Step 3: RAG retrieval
        retrieval_result = self.retriever.retrieve(clean_prompt)
        context = self.retriever.format_context(retrieval_result)

        # Step 4: LLM generates response
        history_str = ""
        if conversation_history:
            for msg in conversation_history[-4:]:
                role = "User" if msg.get("role") == "user" else "Assistant"
                history_str += role + ": " + msg.get("content", "") + "\n"

        response = self.responder.generate(clean_prompt, context, history_str)

        # Step 5: Output firewall
        output_pii = self.output_firewall.scan_pii(response)
        faithfulness = self.output_firewall.score_faithfulness(
            response, retrieval_result.chunks
        )

        if output_pii:
            response = (
                "[Response contained sensitive information and was "
                "redacted by the output firewall. Please rephrase "
                "your question without referencing specific customer data.]"
            )

        # Step 6: Audit log
        log_interaction(
            original_text=scrub_result.original_text,
            redacted_text=clean_prompt,
            entities_found=scrub_result.entities_found,
            redaction_count=scrub_result.redaction_count,
            has_pii=scrub_result.has_pii,
            risk_score=risk_result.score,
            risk_action=risk_result.action,
            faithfulness=faithfulness,
            response=response,
            phase="pipeline"
        )

        return PipelineResult(
            original_prompt=prompt,
            redacted_prompt=clean_prompt,
            pii_found=scrub_result.entities_found,
            pii_count=scrub_result.redaction_count,
            risk_score=risk_result.score,
            risk_action=risk_result.action,
            risk_reason=risk_result.reason,
            sources=retrieval_result.sources,
            chunks_retrieved=retrieval_result.chunk_count,
            response=response,
            faithfulness_score=faithfulness,
            output_pii_found=output_pii,
            blocked=False,
            block_reason=""
        )

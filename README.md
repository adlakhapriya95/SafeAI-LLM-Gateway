# SafeAI LLM Gateway

**AI security and governance middleware for regulated industries.**

SafeAI sits between users and any large language model, intercepting every prompt before the model sees it and every response before the user sees it. It enforces PII detection, threat classification, RAG-grounded responses, and compliance audit logging at every step.

Built for banking, insurance, and healthcare where GDPR, PCI-DSS, HIPAA, and the EU AI Act make uncontrolled LLM deployment a regulatory risk.

---

## Overview

SafeAI is a security middleware layer that addresses a specific and well-documented blocker in enterprise AI adoption. Regulated institutions have the motivation and technical capability to deploy LLMs internally, but cannot obtain compliance sign-off without documented evidence that the system does not expose them to data protection liability.

The architecture is local-first. Every component except the final LLM call runs entirely on-premise. No customer prompt, no document, and no embedding leaves the institution's environment during the classification and retrieval phases. This is the minimum requirement for passing a data residency review at a tier-1 bank.

The design principle is defence-in-depth. No single layer is assumed to be complete. The PII scrubber does not catch everything. The risk scorer does not catch everything. Each layer compensates for the gaps in the others. A prompt must pass all layers before reaching the LLM.

---

## Core Capabilities

### PII Detection and Redaction

Every incoming prompt is scanned for personally identifiable information before the text reaches any processing layer. SafeAI uses Microsoft Presidio as the detection engine, which runs locally using pre-trained NLP models. The default configuration detects over twenty entity types including names, email addresses, phone numbers, credit card numbers, IBAN codes, national insurance numbers, Aadhaar numbers, and bank account references.

Custom regex-based recognisers are added for financial identifiers that Presidio does not cover by default: CUSIP codes, ISIN numbers, internal account reference formats, loan identifiers, and UK National Insurance Numbers. The confidence threshold is set at 0.50. DATE_TIME is excluded from the entity list to prevent false positives on date references in compliance questions.

PII redaction happens before the prompt reaches the risk scorer, the RAG retrieval layer, and the LLM. The original text is preserved only in the audit log for forensic purposes.

### Three-Layer Threat Detection

SafeAI scores every incoming prompt for adversarial intent using a three-layer classifier.

**Layer 1 — Rule-based classifier.** Regular expressions match known attack patterns instantly with no model required. Patterns cover classic prompt injection phrases, jailbreak keywords, admin mode claims, maintenance mode tricks, and context reset instructions. Any match results in an immediate BLOCK at score 0.95.

**Layer 2 — Semantic similarity scorer.** Uses the all-MiniLM-L6-v2 sentence-transformer model. Encodes the incoming prompt into a 384-dimensional vector and measures cosine similarity against a library of over 250 known attack examples across four categories: prompt injection, jailbreaks, social engineering, and data exfiltration. A counter-signal of known safe query examples is subtracted at 0.3 weight to reduce false positives on legitimate compliance questions.

**Layer 3 — LLM judge.** Fires only when the semantic score falls between 0.05 and 0.70. Uses Groq-hosted Llama 3.1 to reason about prompt intent and return a structured risk score, threat type, and reasoning. The final risk score is a weighted combination of 40 percent semantic score and 60 percent LLM judge score.

### RAG-Grounded Responses

Every response is grounded in a vector database of institution-specific regulatory documents. The default document set includes the full text of GDPR Regulation 2016/679, the PCI-DSS Quick Reference Guide v3.2.1, the OWASP Top 10 for LLM Applications 2023, and an internal AI usage policy template.

Each document is chunked into 500-character segments with 50-character overlap and embedded into a local ChromaDB instance. For every prompt that passes the input firewall, the retriever performs a semantic similarity search and returns the top three most relevant chunks. The LLM is instructed to answer only from the provided context.

### Output Firewall

Before the LLM response reaches the user, a second Presidio scan checks the output for PII. If any PII entity is detected with confidence above 0.60, the response is replaced with a redaction notice. Faithfulness scoring measures keyword overlap between the response and the retrieved chunks to detect when the LLM may be drawing on pre-trained knowledge rather than the retrieved context.

### Compliance Audit Trail

Every interaction is written to a local SQLite database with a full record of the original prompt, the redacted version, PII entities found, risk score, risk action, the layer that caught the threat, retrieved document sources, the LLM response, faithfulness score, session identifier, IP address, user agent string, and device fingerprint.

When a prompt is blocked, SafeAI fires a webhook to an N8N automation workflow which dispatches an email alert to the compliance team containing the session ID, risk score, threat type, and truncated redacted prompt.

---

## Architecture

Every prompt travels through eight sequential layers before a response reaches the user.

| Layer | Name | Function |
|-------|------|----------|
| 1 | PII Scrubber | Presidio with custom financial recognisers. 20+ entity types. Confidence threshold 0.50. DATE_TIME excluded. |
| 2 | Risk Scorer | Three-layer classifier: regex rules, semantic similarity, LLM judge. BLOCK above 0.75. FLAG 0.50 to 0.75. PASS below 0.50. |
| 3 | Document Processor | PDF, DOCX, TXT extraction. PyMuPDF for PDFs. LSB variance steganography check on image uploads. |
| 4 | RAG Retrieval | ChromaDB local vector store. Returns top 3 chunks from regulatory document corpus. all-MiniLM-L6-v2 embeddings. |
| 5 | Agentic Router | LangGraph agent routes between document retrieval and tool calls based on prompt intent. |
| 6 | LLM | Groq-hosted Llama 3.1. Temperature 0.1. Never receives original PII. Response grounded in retrieved chunks. |
| 7 | Output Firewall | Second Presidio PII scan on LLM response. Faithfulness scoring via keyword overlap. Response redacted if PII detected. |
| 8 | Audit Log and Escalation | Full interaction written to SQLite with session ID, IP, user agent, device fingerprint. N8N webhook on BLOCK events. |

---

## Performance

Performance was established through systematic batch testing using 165 labelled prompts across seven categories: safe policy queries, prompt injection attacks, jailbreaks, social engineering attempts, data exfiltration requests, PII-containing prompts, and edge cases.

| Metric | Result | Benchmark for regulated deployment |
|--------|--------|--------------------------------------|
| PII detection recall | 100% | Above 95% |
| Threat detection recall | 94.0% | Above 90% |
| False positive rate | 8.3% | Below 15% for internal tools |
| False negative rate | 6.0% | Below 10% for v1 |
| Precision | 97.8% | Above 90% |
| F1 score | 95.2% | Above 90% |
| Average response latency | 3.89 seconds | Below 5 seconds |

### Performance by category

| Category | Prompts | Recall | Notes |
|----------|---------|--------|-------|
| Safe policy queries | 30 | 100% TN | Zero false positives |
| Prompt injection | 30 | 96.7% | 1 false negative |
| Jailbreaks | 30 | 100% | All caught |
| Social engineering | 30 | 100% | All caught |
| Data exfiltration | 20 | 100% | All caught |
| PII in prompts | 15 | 100% TN | All correctly passed with redaction |

---

## Quick Start

```bash
git clone https://github.com/adlakhapriya95/SafeAI-LLM-Gateway.git
cd SafeAI-LLM-Gateway
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the root folder with the following:

```
GROQ_API_KEY=your_groq_api_key_here
N8N_WEBHOOK_URL=your_n8n_webhook_url_here
TESTING=true
```

Load regulatory documents into ChromaDB:

```bash
python load_documents.py
```

Start the app:

```bash
streamlit run app.py
```

In a separate terminal, start N8N for escalation alerts:

```bash
n8n start
```

Open `http://localhost:8501` in your browser.

---

## Regulatory Documents

The following documents should be placed in the `docs/` folder before running `load_documents.py`. They are excluded from the repository due to file size.

- GDPR Regulation 2016/679 full text — available from EUR-Lex
- PCI-DSS Quick Reference Guide v3.2.1 — available from PCI SSC
- OWASP Top 10 for LLM Applications 2023 — available from owasp.org
- Internal AI usage policy — template provided in `docs/internal_ai_policy.txt`

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| PII detection | Microsoft Presidio |
| Semantic scoring | sentence-transformers all-MiniLM-L6-v2 |
| LLM judge | Groq API, Llama 3.1 8B Instant |
| Vector database | ChromaDB local |
| LLM framework | LangChain, LangGraph |
| Document processing | PyMuPDF, python-docx |
| Escalation workflow | N8N self-hosted |
| User interface | Streamlit |
| Audit storage | SQLite |

---

## Regulatory Coverage

| Framework | How SafeAI addresses it |
|-----------|------------------------|
| GDPR Article 32 | PII scrubbing before processing. Audit log for data subject rights. Session tracking for breach investigation. |
| GDPR Article 5 | Data minimisation enforced by PII redaction. Responses grounded in approved policy documents only. |
| PCI-DSS Requirement 3.4 | Credit card, IBAN, and account number redaction at input and output. Custom Presidio recognisers. |
| PCI-DSS Requirement 6.3 | Adversarial input testing via 165-prompt batch test suite and weekly automated agent. |
| HIPAA Security Rule 164.312 | PHI entity types in Presidio detection. Audit trail with timestamps and user fingerprinting. |
| EU AI Act Article 9 | Continuous risk management system with documented adversarial testing history. |
| EU AI Act Article 10 | Training data documented in KNOWN_ATTACKS with timestamps. Batch test validation in audit backups. |
| OWASP LLM Top 10 | Prompt injection covered by Layers 1 and 2. Insecure output handled by output firewall. |
| FCA Model Risk Guidance | Audit trail, pipeline trace for explainability, faithfulness scoring, session analysis. |

---

## Automated Improvement Agent

SafeAI includes a self-improving adversarial agent that runs on a weekly schedule via cron job.

Each iteration the agent:
1. Measures a baseline false negative rate on a held-out benchmark of 24 unseen attack prompts
2. Generates 12 novel attack variants per threat category using the LLM at temperature 0.85
3. Tests each variant against the live pipeline
4. Adds confirmed slip-throughs to KNOWN_ATTACKS
5. Rebuilds semantic embeddings
6. Re-measures on the held-out benchmark and reports the before and after delta

The held-out benchmark prompts are never added to the training library. This ensures the benchmark measures genuine generalisation to unseen attacks rather than memorisation.

To run manually:

```bash
python agent_improve.py 1
```

To schedule weekly:

```bash
crontab -e
# Add: 0 2 * * 0 cd ~/Desktop/SafeAI && source venv/bin/activate && python agent_improve.py >> audit/agent_log.txt 2>&1
```

---

## Known Limitations

- **Social engineering ceiling.** Attacks that closely resemble legitimate internal email communication cannot be reliably detected by semantic similarity alone. Requires user authentication and role-based anomaly detection in v2.
- **Semantic layer ceiling.** The sentence-transformer has a geometric limit. Adding more attack examples improves coverage but can tighten the detection boundary and increase false positives. V2 fix: fine-tuned intent classifier.
- **Scanned images.** PyMuPDF only reads digital text layers. OCR via Tesseract is available on Linux production servers.
- **External LLM call.** The Groq API sends prompts externally. Replaceable with Ollama for full on-prem deployment in one environment variable change.
- **No cross-session memory.** Each session begins from scratch. V2 fix: session-level cumulative risk scoring.

---

## Project Documentation

Full technical documentation is available in the repository:

- `README.docx` — Extended overview with full architecture detail
- `ARCHITECTURE.docx` — Phase-by-phase build narrative, threshold decisions, testing methodology
- `SafeAI_PRD_v2.docx` — Full product requirements document with regulatory mapping and glossary

---

## Author

Priya Adlakha — Product Manager  
Background in financial data and AI governance.  
Built as a portfolio project targeting AI PM roles in regulated fintech.

[LinkedIn](https://linkedin.com/in/priyaadlakha) | [GitHub](https://github.com/adlakhapriya95)

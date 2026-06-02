"""
Kamishield — Phase 1: PII Scrubber
-----------------------------------
Scans incoming text for personally identifiable information
and replaces each detected entity with a typed placeholder.

The original text is NEVER forwarded. Only the redacted
version moves further down the pipeline.

Libraries used:
    presidio-analyzer   detects PII entities with confidence scores
    presidio-anonymizer replaces detected entities with placeholders
"""

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_analyzer import PatternRecognizer, Pattern
import logging
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(level=logging.WARNING)


# ─────────────────────────────────────────────
# Output data structure
# ─────────────────────────────────────────────

@dataclass
class ScrubResult:
    original_text: str          # never forwarded — stored locally only
    redacted_text: str          # this moves to the next pipeline layer
    entities_found: list        # list of detected entity types
    redaction_count: int        # how many replacements were made
    has_pii: bool               # quick boolean flag


# ─────────────────────────────────────────────
# Custom financial recognisers
# These are domain-specific patterns Presidio
# does not know about by default.
# ─────────────────────────────────────────────

def build_custom_recognisers():
    """
    Returns a list of custom pattern recognisers
    for financial-industry-specific identifiers.
    """

    # CUSIP — US securities identifier (9 characters)
    cusip = PatternRecognizer(
        supported_entity="CUSIP",
        patterns=[Pattern(
            name="cusip_pattern",
            regex=r"\b[0-9]{3}[A-Z0-9]{5}[0-9]\b",
            score=0.85
        )],
        context=["cusip", "security", "bond", "equity"]
    )

    # ISIN — International Securities Identification Number
    isin = PatternRecognizer(
        supported_entity="ISIN",
        patterns=[Pattern(
            name="isin_pattern",
            regex=r"\b[A-Z]{2}[A-Z0-9]{9}[0-9]\b",
            score=0.85
        )],
        context=["isin", "security", "international"]
    )

    # Internal account reference (format: ACC-YYYY-NNNNN)
    account_ref = PatternRecognizer(
        supported_entity="ACCOUNT_REF",
        patterns=[Pattern(
            name="account_ref_pattern",
            regex=r"\bACC[-/][0-9]{4}[-/][0-9]{4,8}\b",
            score=0.90
        )],
        context=["account", "reference", "acc"]
    )

    # Loan ID (format: LN/YYYY/NNNNN)
    loan_id = PatternRecognizer(
        supported_entity="LOAN_ID",
        patterns=[Pattern(
            name="loan_id_pattern",
            regex=r"\bLN[-/][0-9]{4}[-/][0-9]{4,8}\b",
            score=0.90
        )],
        context=["loan", "lending", "credit"]
    )

    # UK National Insurance Number
    uk_nin = PatternRecognizer(
        supported_entity="UK_NIN",
        patterns=[Pattern(
            name="uk_nin_pattern",
            regex=r"\b[A-Z]{2}[0-9]{6}[A-D]\b",
            score=0.80
        )],
        context=["national insurance", "NI number", "NINO"]
    )

    # Indian Aadhaar Number (12 digits, sometimes spaced)
    aadhaar = PatternRecognizer(
        supported_entity="AADHAAR",
        patterns=[Pattern(
            name="aadhaar_pattern",
            regex=r"\b[2-9][0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b",
            score=0.80
        )],
        context=["aadhaar", "uid", "unique identification"]
    )

    return [cusip, isin, account_ref, loan_id, uk_nin, aadhaar]


# ─────────────────────────────────────────────
# Core scrubber class
# ─────────────────────────────────────────────

class PIIScrubber:
    """
    Wraps Presidio's analyzer and anonymizer with
    custom financial recognisers and sensible defaults
    for a regulated industry context.
    """

    # Entity types we actively scan for.
    # Standard Presidio types + our custom financial ones.
    ENTITIES_TO_DETECT = [
        # Standard PII
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "LOCATION",

        # Financial / identity
        "CREDIT_CARD",
        "IBAN_CODE",
        "US_SSN",
        "US_BANK_NUMBER",
        "US_PASSPORT",
        "US_DRIVER_LICENSE",
        "UK_NHS",
        "SG_NRIC_FIN",
        "AU_ACN",
        "AU_TFN",
        "AU_ABN",
        "IN_PAN",

        # Our custom financial types
        "CUSIP",
        "ISIN",
        "ACCOUNT_REF",
        "LOAN_ID",
        "UK_NIN",
        "AADHAAR",
    ]

    def __init__(self, confidence_threshold: float = 0.60):
        """
        confidence_threshold: minimum Presidio confidence score
        for an entity to be redacted. 0.6 is deliberately
        conservative — better to redact a false positive than
        miss a real PII entity in a banking context.
        """
        self.confidence_threshold = confidence_threshold
        self._setup_engines()

    def _setup_engines(self):
        """Initialise Presidio analyzer and anonymizer."""

        # Build the NLP engine (uses spaCy en_core_web_lg)
        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}]
        }
        provider = NlpEngineProvider(nlp_configuration=nlp_config)
        nlp_engine = provider.create_engine()

        # Build the analyzer and add our custom recognisers
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        for recogniser in build_custom_recognisers():
            self.analyzer.registry.add_recognizer(recogniser)

        # Build the anonymizer
        self.anonymizer = AnonymizerEngine()

    def scrub(self, text: str, language: str = "en") -> ScrubResult:
        """
        Main entry point. Takes raw text, returns a ScrubResult
        with the redacted version and a log of what was found.

        The original text is kept in ScrubResult.original_text
        for local audit logging only. It is never returned
        to the user or forwarded to the LLM.
        """

        if not text or not text.strip():
            return ScrubResult(
                original_text=text,
                redacted_text=text,
                entities_found=[],
                redaction_count=0,
                has_pii=False
            )

        # Step 1: Detect PII entities
        detected = self.analyzer.analyze(
            text=text,
            entities=self.ENTITIES_TO_DETECT,
            language=language,
            score_threshold=self.confidence_threshold
        )

        if not detected:
            return ScrubResult(
                original_text=text,
                redacted_text=text,
                entities_found=[],
                redaction_count=0,
                has_pii=False
            )

        # Step 2: Build replacement operators
        # Each entity type gets replaced with <ENTITY_TYPE>
        operators = {}
        for result in detected:
            operators[result.entity_type] = OperatorConfig(
                "replace",
                {"new_value": f"<{result.entity_type}>"}
            )

        # Step 3: Anonymize — replace detected entities
        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=detected,
            operators=operators
        )

        entity_types = sorted(set(r.entity_type for r in detected))

        return ScrubResult(
            original_text=text,
            redacted_text=anonymized.text,
            entities_found=entity_types,
            redaction_count=len(detected),
            has_pii=True
        )

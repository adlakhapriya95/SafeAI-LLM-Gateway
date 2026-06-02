"""
SafeAI — Phase 5: OCR Image Scanner
--------------------------------------
Accepts image uploads and extracts all text before
anything reaches the LLM.

Three passes run on every image:

Pass 1 — Standard OCR
    Tesseract reads the image as-is and extracts
    all visible text.

Pass 2 — Enhanced OCR
    Image converted to high-contrast grayscale.
    Tesseract re-runs to catch text invisible to
    the naked eye (white-on-white, micro-font).

Pass 3 — Steganography check
    Statistical analysis of pixel least-significant
    bits. Natural images have high LSB variance.
    Steganographic images show abnormally low variance
    because hidden data regularises the pixel values.

Output is plain text that enters Phase 1 (PII scrubber)
and Phase 2 (risk scorer) exactly like a typed prompt.
The image itself never reaches the LLM.

Libraries used:
    pytesseract   Python wrapper for Tesseract OCR engine
    Pillow        Image loading and processing
    numpy         Pixel-level statistical analysis
"""

import os
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

try:
    import pytesseract
    from PIL import Image, ImageFilter, ImageEnhance
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Windows only — set Tesseract path if specified in .env
TESSERACT_PATH = os.getenv("TESSERACT_PATH", "")
if TESSERACT_PATH and OCR_AVAILABLE:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ─────────────────────────────────────────────
# Output data structure
# ─────────────────────────────────────────────

@dataclass
class OCRResult:
    file_path: str
    visible_text: str           # text from standard OCR pass
    hidden_text: str            # additional text from enhanced pass
    combined_text: str          # everything merged for pipeline input
    hidden_content_detected: bool   # was extra text found in pass 2
    steganography_suspected: bool   # did pixel analysis flag anomaly
    lsb_variance: float         # raw pixel variance score
    threat_summary: str         # human readable summary


# ─────────────────────────────────────────────
# OCR Scanner
# ─────────────────────────────────────────────

class OCRScanner:
    """
    Extracts text from images using Tesseract OCR
    with enhanced passes for hidden content detection.
    """

    # If enhanced pass finds this many more characters
    # than standard pass, flag as hidden content detected
    HIDDEN_CONTENT_THRESHOLD = 20

    # LSB variance below this value suggests steganography
    # Natural images typically score above 0.20
    STEGANOGRAPHY_THRESHOLD = 0.15

    def __init__(self):
        if not OCR_AVAILABLE:
            raise ImportError(
                "pytesseract or Pillow not installed.\n"
                "Run: pip install pytesseract Pillow"
            )

    def scan(self, image_path: str) -> OCRResult:
        """
        Main entry point. Accepts a file path to any
        image or scanned PDF page.
        Returns OCRResult with all extracted text.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path)

        # Convert to RGB if needed (handles PNG with alpha channel)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # ── Pass 1: Standard OCR ──
        visible_text = self._standard_ocr(img)

        # ── Pass 2: Enhanced OCR for hidden text ──
        hidden_text = self._enhanced_ocr(img)

        # Find text that enhanced pass found but standard missed
        visible_words = set(visible_text.lower().split())
        hidden_words = set(hidden_text.lower().split())
        new_words = hidden_words - visible_words
        extra_chars = len(hidden_text) - len(visible_text)

        hidden_content_detected = (
            extra_chars > self.HIDDEN_CONTENT_THRESHOLD
            and len(new_words) > 3
        )

        # ── Pass 3: Steganography check ──
        lsb_variance, steg_suspected = self._steganography_check(img)

        # Combine all extracted text for pipeline input
        if hidden_content_detected:
            combined = visible_text + " " + hidden_text
        else:
            combined = visible_text

        # Build human readable threat summary
        threats = []
        if hidden_content_detected:
            threats.append(
                f"Hidden text detected: {extra_chars} extra characters "
                f"found in enhanced pass"
            )
        if steg_suspected:
            threats.append(
                f"Steganography suspected: LSB variance {lsb_variance:.3f} "
                f"below threshold {self.STEGANOGRAPHY_THRESHOLD}"
            )

        if threats:
            threat_summary = " | ".join(threats)
        else:
            threat_summary = "No image-level threats detected"

        return OCRResult(
            file_path=str(path),
            visible_text=visible_text.strip(),
            hidden_text=hidden_text.strip(),
            combined_text=combined.strip(),
            hidden_content_detected=hidden_content_detected,
            steganography_suspected=steg_suspected,
            lsb_variance=lsb_variance,
            threat_summary=threat_summary
        )

    def _standard_ocr(self, img: Image.Image) -> str:
        """
        Standard Tesseract OCR on the original image.
        Returns extracted text as a string.
        """
        try:
            text = pytesseract.image_to_string(img)
            return text
        except Exception as e:
            return f"OCR extraction failed: {str(e)}"

    def _enhanced_ocr(self, img: Image.Image) -> str:
        """
        High-contrast enhanced OCR pass.
        Converts image to grayscale, maximises contrast,
        then re-runs Tesseract.

        This surfaces text that was invisible at normal
        contrast — white text on white background,
        light grey on white, micro-font text.
        """
        try:
            # Convert to greyscale
            grey = img.convert("L")

            # Maximise contrast
            enhancer = ImageEnhance.Contrast(grey)
            high_contrast = enhancer.enhance(3.0)

            # Apply sharpening to improve OCR accuracy
            sharpened = high_contrast.filter(ImageFilter.SHARPEN)

            # Threshold — convert to pure black and white
            # Text becomes black, background becomes white
            bw = sharpened.point(lambda x: 0 if x < 128 else 255, "1")

            text = pytesseract.image_to_string(bw)
            return text
        except Exception as e:
            return ""

    def _steganography_check(self, img: Image.Image) -> tuple:
        """
        Analyses the least-significant bits of pixel values.

        In natural images, LSBs are essentially random
        because they represent tiny variations in colour.
        High variance is expected.

        When steganographic tools hide data in LSBs,
        they overwrite those random bits with structured
        data. This reduces variance in a measurable way.

        Returns (variance_score, is_suspected).
        """
        try:
            img_rgb = img.convert("RGB")
            pixels = np.array(img_rgb)

            # Extract LSB of each colour channel
            lsb_layer = pixels & 1

            # Calculate variance across all LSBs
            variance = float(np.var(lsb_layer))

            suspected = variance < self.STEGANOGRAPHY_THRESHOLD
            return variance, suspected

        except Exception:
            # If analysis fails, do not flag — avoid false positives
            return 1.0, False

"""
MetraSight — AI Legal-Metrology Inspector
Smart India Hackathon 2026 | Problem Statement SIH26034
Ministry of Consumer Affairs, Food & Public Distribution

Dual-strategy compliance pipeline:
  1. PRIMARY  — OpenCV pre-processing + EasyOCR extraction + RegEx/Levenshtein rule engine
  2. FALLBACK — Optional Claude Vision structural parser for fields the primary pipeline
                cannot find or is not confident about (user supplies their own Anthropic API key)

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud (see requirements.txt)

IMPORTANT COMPLIANCE DISCLAIMER
--------------------------------
This tool is a hackathon prototype / decision-support aid. The numeral-height table used in the
Rule 7 (PDP) check below is entered from the author's best recollection of the Legal Metrology
(Packaged Commodities) Rules, 2011 Second Schedule and has NOT been verified against the current
official gazette notification at the time of writing. Before using this application for a real
inspection or any enforcement decision, verify every threshold in `PDP_HEIGHT_TABLE_MM2` against
the official rule text. Do not treat any verdict from this app as legal advice.
"""

import base64
import csv
import io
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# EasyOCR is a heavy import (loads torch) — done lazily inside a cached factory so the
# Streamlit app boots fast and only pays the cost when the user actually needs OCR.
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

# Anthropic SDK is optional — only needed if the user enables the Claude Vision fallback.
try:
    import anthropic
    ANTHROPIC_SDK_AVAILABLE = True
except Exception:
    ANTHROPIC_SDK_AVAILABLE = False


# ============================================================================================
# CONSTANTS — RULE ENGINE CONFIGURATION
# ============================================================================================

STANDARD_UNIT_SYMBOLS = {"g", "kg", "ml", "l", "m", "cm", "n", "pcs"}

# Non-standard tokens that must be FLAGGED as a Rule 6(1)(c) violation even though they are
# semantically the same unit. Key = normalized lowercase token seen in OCR text,
# Value = the standard symbol it should have been.
NON_STANDARD_UNIT_MAP = {
    "gms": "g", "gm": "g", "grams": "g", "gram": "g",
    "kilo": "kg", "kilos": "kg", "kgs": "kg",
    "ltr": "l", "ltrs": "l", "litre": "l", "litres": "l", "liter": "l", "liters": "l",
    "mls": "ml", "millilitre": "ml", "millilitres": "ml",
    "mtr": "m", "mtrs": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "cms": "cm", "centimeter": "cm", "centimeters": "cm", "centimetre": "cm", "centimetres": "cm",
    "newton": "n", "newtons": "n",
    "piece": "pcs", "pieces": "pcs", "pc": "pcs", "nos": "pcs", "no.": "pcs",
}

MONTH_NAMES = (
    "jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    "aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)

# Rule 7 / Second Schedule — indicative numeral-height table (mm^2 area -> minimum numeral
# height in mm). SEE DISCLAIMER AT TOP OF FILE — verify against the official rule text.
PDP_HEIGHT_TABLE_MM2 = [
    (500, 1.0),
    (2500, 2.0),
    (10000, 4.0),
    (50000, 6.0),
    (float("inf"), 8.0),
]

RULE_CITATIONS = {
    "manufacturer": "Rule 6(1)(a) — Name & complete address of Manufacturer/Packer/Importer",
    "country_of_origin": "Rule 6(1)(aa) — Country of Origin (mandatory for imported goods)",
    "commodity_name": "Rule 6(1)(b) — Generic/Common name of the commodity",
    "net_quantity": "Rule 6(1)(c) — Net Quantity in statutory units",
    "mfg_date": "Rule 6(1)(d) — Month & Year of Manufacture/Packing",
    "mrp": "Rule 6(1)(e) — Maximum Retail Price, inclusive of all taxes",
    "usp": "Rule 6(1)(l) — Unit Sale Price (for >1kg/1L or multi-unit packs)",
    "consumer_care": "Rule 6(2) — Consumer Care details (name, address, phone/email)",
    "pdp_numerals": "Rule 7, Table I (Second Schedule) — PDP area vs minimum numeral height",
}

DEFAULT_CLAUDE_MODEL = "claude-sonnet-5"  # change in the sidebar if your account uses a different model string


# ============================================================================================
# DATA MODEL
# ============================================================================================

@dataclass
class FieldResult:
    field_key: str
    label: str
    status: str  # "COMPLIANT" | "NON_COMPLIANT" | "NULL"
    value: str = ""
    confidence: float = 0.0
    source: str = ""            # which uploaded image / panel the evidence came from
    detail: str = ""            # human-readable explanation
    rescan_hint: str = ""       # what to re-photograph if NULL
    rule: str = ""


@dataclass
class EvidenceStore:
    """Aggregated OCR evidence pooled across every uploaded image of one product."""
    raw_tokens: list = field(default_factory=list)   # list of dict: text, confidence, source, bbox
    full_text: str = ""
    per_image_results: dict = field(default_factory=dict)  # filename -> list of ocr rows


# ============================================================================================
# LEVENSHTEIN DISTANCE (pure python — no extra dependency needed on Streamlit Cloud)
# ============================================================================================

def levenshtein(a: str, b: str) -> int:
    a, b = a.lower(), b.lower()
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur_row = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            insert_cost = cur_row[j - 1] + 1
            delete_cost = prev_row[j] + 1
            replace_cost = prev_row[j - 1] + (0 if ca == cb else 1)
            cur_row[j] = min(insert_cost, delete_cost, replace_cost)
        prev_row = cur_row
    return prev_row[-1]


def fuzzy_ratio(a: str, b: str) -> float:
    """0..1 similarity based on normalized Levenshtein distance."""
    a, b = a.lower().strip(), b.lower().strip()
    max_len = max(len(a), len(b), 1)
    return 1.0 - (levenshtein(a, b) / max_len)


def fuzzy_contains(haystack: str, keywords: list, threshold: float = 0.78) -> Optional[str]:
    """Search a block of text for fuzzy matches to any keyword; return the matched keyword or None."""
    haystack_lower = haystack.lower()
    words = re.findall(r"[a-zA-Z\.]{2,}", haystack_lower)
    for kw in keywords:
        if kw.lower() in haystack_lower:
            return kw
        for w in words:
            if fuzzy_ratio(w, kw) >= threshold:
                return kw
    return None


# ============================================================================================
# IMAGE PRE-PROCESSING ENGINE (OpenCV / PIL)
# ============================================================================================

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2_to_pil(arr: np.ndarray) -> Image.Image:
    if len(arr.shape) == 2:
        return Image.fromarray(arr)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def apply_adaptive_threshold(gray: np.ndarray, block_size: int = 25, c: int = 10) -> np.ndarray:
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, h=10)


def preprocess_pipeline(
    pil_img: Image.Image,
    use_grayscale: bool = True,
    use_clahe: bool = True,
    use_denoise: bool = True,
    use_adaptive_threshold: bool = False,
) -> Image.Image:
    """
    Cleans up glossy / low-contrast / curved packaging photos before OCR.
    Returns a PIL image (RGB or L) ready for EasyOCR.
    """
    bgr = pil_to_cv2(pil_img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if use_grayscale else bgr

    if use_grayscale:
        if use_denoise:
            gray = denoise(gray)
        if use_clahe:
            gray = apply_clahe(gray)
        if use_adaptive_threshold:
            gray = apply_adaptive_threshold(gray)
        return cv2_to_pil(gray)

    return cv2_to_pil(gray)


# ============================================================================================
# OCR ENGINE
# ============================================================================================

@st.cache_resource(show_spinner="Loading EasyOCR model (first run only, ~1-2 min)...")
def get_ocr_reader():
    if not EASYOCR_AVAILABLE:
        return None
    return easyocr.Reader(["en"], gpu=False)


def run_ocr(pil_img: Image.Image, reader) -> list:
    """
    Returns list of dicts: {text, confidence (0-1), bbox (4 points), height_px}
    """
    if reader is None:
        return []
    arr = np.array(pil_img.convert("RGB"))
    results = reader.readtext(arr)
    rows = []
    for bbox, text, conf in results:
        ys = [p[1] for p in bbox]
        height_px = max(ys) - min(ys)
        rows.append({
            "text": text,
            "confidence": float(conf),
            "bbox": bbox,
            "height_px": float(height_px),
        })
    return rows


def draw_bounding_boxes(pil_img: Image.Image, ocr_rows: list, min_conf_highlight: float = 0.5) -> Image.Image:
    bgr = pil_to_cv2(pil_img)
    for row in ocr_rows:
        pts = np.array(row["bbox"], dtype=np.int32).reshape((-1, 1, 2))
        color = (0, 200, 0) if row["confidence"] >= min_conf_highlight else (0, 0, 230)
        cv2.polylines(bgr, [pts], isClosed=True, color=color, thickness=2)
    return cv2_to_pil(bgr)


# ============================================================================================
# RULE ENGINE — FIELD EXTRACTORS
# Each extractor scans the pooled evidence (raw OCR tokens across ALL uploaded images of the
# product) using RegEx first, then falls back to fuzzy/Levenshtein keyword search for anchor
# phrases. Every extractor returns (value, confidence, source_image, detail) or None.
# ============================================================================================

def _best_source_for_text(store: EvidenceStore, needle: str) -> str:
    needle_lower = needle.lower()[:25]
    for tok in store.raw_tokens:
        if needle_lower and needle_lower in tok["text"].lower():
            return tok["source"]
    return store.raw_tokens[0]["source"] if store.raw_tokens else "unknown"


def extract_country_of_origin(store: EvidenceStore) -> Optional[dict]:
    pattern = re.compile(
        r"(?:country\s+of\s+origin|made\s+in|manufactured\s+in|product\s+of|origin\s*:)\s*[:\-]?\s*([A-Za-z\s]{3,30})",
        re.IGNORECASE,
    )
    m = pattern.search(store.full_text)
    if m:
        country = m.group(1).strip().split("\n")[0][:30]
        conf = _avg_conf_near(store, m.group(0))
        return {"value": country, "confidence": conf, "source": _best_source_for_text(store, m.group(0))}
    kw = fuzzy_contains(store.full_text, ["country of origin", "made in india", "product of india"])
    if kw:
        conf = 0.45
        return {"value": kw, "confidence": conf, "source": _best_source_for_text(store, kw)}
    return None


def extract_manufacturer(store: EvidenceStore) -> Optional[dict]:
    anchors = ["manufactured by", "marketed by", "packed by", "manufacturer", "packer", "importer", "mfd by", "mkt by"]
    pattern = re.compile(
        r"(?:manufactured\s+by|marketed\s+by|packed\s+by|mfd\s+by|mkt\s+by|manufacturer|packer|importer)\s*[:\-]?\s*(.{8,120})",
        re.IGNORECASE,
    )
    m = pattern.search(store.full_text)
    if m:
        addr = re.split(r"\n{2,}|(?:pin\s*code)", m.group(1), maxsplit=1)[0].strip()
        conf = _avg_conf_near(store, m.group(0))
        return {"value": addr[:200], "confidence": conf, "source": _best_source_for_text(store, m.group(0))}
    kw = fuzzy_contains(store.full_text, anchors)
    if kw:
        return {"value": f"(anchor '{kw}' found, address text unclear)", "confidence": 0.4,
                "source": _best_source_for_text(store, kw)}
    return None


def extract_commodity_name(store: EvidenceStore) -> Optional[dict]:
    # Heuristic: the highest-confidence, largest-height token near the top of the front panel
    # that is NOT a pure number/unit/date is usually the product/common name. We use the
    # single highest confidence alphabetic token block as a best-effort common name.
    candidates = [
        t for t in store.raw_tokens
        if re.search(r"[A-Za-z]{3,}", t["text"]) and not re.search(r"\d{2,}", t["text"])
        and len(t["text"]) <= 40
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t["confidence"], t["height_px"]), reverse=True)
    top = candidates[0]
    return {"value": top["text"], "confidence": top["confidence"], "source": top["source"]}


def extract_net_quantity(store: EvidenceStore) -> Optional[dict]:
    pattern = re.compile(
        r"(?:net\s*(?:wt|weight|qty|quantity|contents)?\s*[:\-]?\s*)?(\d+(?:\.\d+)?)\s*"
        r"(g|gm|gms|grams?|kgs?|kilos?|ml|mls|l|ltrs?|litres?|liters?|m|mtrs?|meters?|metres?|"
        r"cm|cms|centimeters?|centimetres?|n|newtons?|pcs|pc|pieces?|nos)\b",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(store.full_text))
    if not matches:
        return None
    m = matches[0]
    number, raw_unit = m.group(1), m.group(2).lower()
    conf = _avg_conf_near(store, m.group(0))

    if raw_unit in STANDARD_UNIT_SYMBOLS:
        return {
            "value": f"{number} {raw_unit}", "confidence": conf,
            "source": _best_source_for_text(store, m.group(0)),
            "violation": False, "raw_unit": raw_unit, "number": number,
        }
    normalized = NON_STANDARD_UNIT_MAP.get(raw_unit)
    return {
        "value": f"{number} {raw_unit}", "confidence": conf,
        "source": _best_source_for_text(store, m.group(0)),
        "violation": True, "raw_unit": raw_unit, "number": number,
        "expected_symbol": normalized or "?",
    }


def extract_mfg_date(store: EvidenceStore) -> Optional[dict]:
    numeric_pattern = re.compile(r"\b(0?[1-9]|1[0-2])\s*[\/\-\.]\s*((?:19|20)\d{2})\b")
    text_pattern = re.compile(rf"\b({MONTH_NAMES})[\s,\.]*((?:19|20)\d{{2}})\b", re.IGNORECASE)
    label_hint = re.search(r"(mfg|manufactur\w*|pkd|packed|packing)\s*(?:date|dt)?\s*[:\-]?", store.full_text, re.IGNORECASE)

    m = numeric_pattern.search(store.full_text) or text_pattern.search(store.full_text)
    if m:
        conf = _avg_conf_near(store, m.group(0))
        if label_hint:
            conf = min(1.0, conf + 0.1)
        return {"value": m.group(0).strip(), "confidence": conf, "source": _best_source_for_text(store, m.group(0))}
    return None


def extract_mrp(store: EvidenceStore) -> Optional[dict]:
    pattern = re.compile(
        r"(?:mrp|m\.r\.p\.?|maximum\s+retail\s+price)\s*[:\-]?\s*(?:rs\.?|inr|₹)?\s*(\d+(?:[.,]\d{1,2})?)",
        re.IGNORECASE,
    )
    m = pattern.search(store.full_text)
    if not m:
        # look for a bare currency figure as a weaker fallback
        m = re.search(r"(?:₹|rs\.?)\s*(\d+(?:[.,]\d{1,2})?)", store.full_text, re.IGNORECASE)
        if not m:
            return None
    price = m.group(1)
    conf = _avg_conf_near(store, m.group(0))

    tax_kw = fuzzy_contains(
        store.full_text,
        ["inclusive of all taxes", "incl. of all taxes", "incl of all taxes", "inclusive of taxes"],
        threshold=0.72,
    )
    return {
        "value": f"Rs. {price}", "confidence": conf,
        "source": _best_source_for_text(store, m.group(0)),
        "has_tax_clause": bool(tax_kw),
    }


def extract_usp(store: EvidenceStore) -> Optional[dict]:
    pattern = re.compile(
        r"(?:unit\s+sale\s+price|usp)\s*[:\-]?\s*(?:rs\.?|inr|₹)?\s*(\d+(?:[.,]\d{1,2})?)\s*(?:per|/)\s*"
        r"(kg|g|l|ml|litre|liter|gram|piece|pcs)?",
        re.IGNORECASE,
    )
    m = pattern.search(store.full_text)
    if m:
        conf = _avg_conf_near(store, m.group(0))
        return {"value": m.group(0).strip(), "confidence": conf, "source": _best_source_for_text(store, m.group(0))}
    return None


def extract_consumer_care(store: EvidenceStore) -> Optional[dict]:
    phone_pattern = re.compile(r"(?:\+91[\-\s]?)?\b[6-9]\d{9}\b|\b1800[\-\s]?\d{2,3}[\-\s]?\d{3,4}\b")
    email_pattern = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
    care_kw = fuzzy_contains(store.full_text, ["consumer care", "customer care", "customer support", "care of consumer"])

    phone = phone_pattern.search(store.full_text)
    email = email_pattern.search(store.full_text)

    if not (phone or email or care_kw):
        return None

    parts = []
    conf_components = []
    if care_kw:
        parts.append(f"anchor:'{care_kw}'")
        conf_components.append(0.5)
    if phone:
        parts.append(f"phone:{phone.group(0)}")
        conf_components.append(_avg_conf_near(store, phone.group(0)))
    if email:
        parts.append(f"email:{email.group(0)}")
        conf_components.append(_avg_conf_near(store, email.group(0)))

    conf = sum(conf_components) / len(conf_components) if conf_components else 0.4
    source_text = phone.group(0) if phone else (email.group(0) if email else care_kw)
    return {"value": " | ".join(parts), "confidence": conf, "source": _best_source_for_text(store, source_text),
            "has_phone": bool(phone), "has_email": bool(email)}


def _avg_conf_near(store: EvidenceStore, snippet: str) -> float:
    """Approximate confidence for a regex match by averaging confidences of OCR tokens whose
    text overlaps with the matched snippet. Falls back to the mean confidence of all tokens."""
    snippet_lower = snippet.lower()
    hits = [t["confidence"] for t in store.raw_tokens if t["text"] and t["text"].lower() in snippet_lower
            or snippet_lower[:12] in t["text"].lower()]
    if hits:
        return sum(hits) / len(hits)
    if store.raw_tokens:
        return sum(t["confidence"] for t in store.raw_tokens) / len(store.raw_tokens)
    return 0.5


# ============================================================================================
# PDP / RULE 7 NUMERAL HEIGHT CHECK
# ============================================================================================

def required_numeral_height_mm(pdp_area_cm2: float) -> float:
    area_mm2 = pdp_area_cm2 * 100.0
    for threshold, height in PDP_HEIGHT_TABLE_MM2:
        if area_mm2 <= threshold:
            return height
    return PDP_HEIGHT_TABLE_MM2[-1][1]


def check_pdp_numerals(store: EvidenceStore, pdp_area_cm2: float, px_per_mm: float) -> FieldResult:
    required_mm = required_numeral_height_mm(pdp_area_cm2)
    if px_per_mm <= 0 or not store.raw_tokens:
        return FieldResult(
            field_key="pdp_numerals", label="PDP Numeral Height", status="NULL",
            detail="No calibration (px/mm) or no OCR tokens available to measure numeral height.",
            rescan_hint="Provide a pixels-per-mm calibration value and ensure at least one image has readable text.",
            rule=RULE_CITATIONS["pdp_numerals"],
        )

    # Only consider tokens that look like the numerals used for MRP / Net Qty declarations
    numeral_tokens = [t for t in store.raw_tokens if re.search(r"\d", t["text"])]
    if not numeral_tokens:
        return FieldResult(
            field_key="pdp_numerals", label="PDP Numeral Height", status="NULL",
            detail="No numeral text detected on the Principal Display Panel images.",
            rescan_hint="Upload a sharper, well-lit close-up of the PDP showing MRP / Net Quantity numerals.",
            rule=RULE_CITATIONS["pdp_numerals"],
        )

    heights_mm = [t["height_px"] / px_per_mm for t in numeral_tokens]
    min_found = min(heights_mm)
    avg_conf = sum(t["confidence"] for t in numeral_tokens) / len(numeral_tokens)

    if min_found + 1e-6 >= required_mm:
        return FieldResult(
            field_key="pdp_numerals", label="PDP Numeral Height", status="COMPLIANT",
            value=f"{min_found:.2f} mm (min required {required_mm:.1f} mm)",
            confidence=avg_conf, source="calculated from OCR bounding boxes",
            detail=f"Smallest detected numeral height {min_found:.2f} mm meets the {required_mm:.1f} mm minimum "
                   f"for a PDP area of {pdp_area_cm2:.1f} cm².",
            rule=RULE_CITATIONS["pdp_numerals"],
        )
    return FieldResult(
        field_key="pdp_numerals", label="PDP Numeral Height", status="NON_COMPLIANT",
        value=f"{min_found:.2f} mm (min required {required_mm:.1f} mm)",
        confidence=avg_conf, source="calculated from OCR bounding boxes",
        detail=f"Smallest detected numeral height {min_found:.2f} mm is BELOW the {required_mm:.1f} mm minimum "
               f"required for a PDP area of {pdp_area_cm2:.1f} cm².",
        rule=RULE_CITATIONS["pdp_numerals"],
    )


# ============================================================================================
# CLAUDE VISION FALLBACK (optional, user-supplied API key)
# ============================================================================================

FALLBACK_FIELD_PROMPTS = {
    "manufacturer": "the full name and complete address of the manufacturer, packer, or importer",
    "country_of_origin": "the declared country of origin",
    "commodity_name": "the generic/common name of the product",
    "net_quantity": "the net quantity value and its unit exactly as printed",
    "mfg_date": "the month and year of manufacture or packing",
    "mrp": "the Maximum Retail Price figure and whether an 'inclusive of all taxes' phrase appears near it",
    "usp": "the Unit Sale Price if printed (price per kg/litre/unit)",
    "consumer_care": "any consumer/customer care name, address, phone number, or email",
}


def call_claude_vision_fallback(api_key: str, model: str, image_bytes: bytes, missing_field_key: str) -> Optional[dict]:
    """
    Sends ONE image to Claude with a vision request asking specifically for the missing field.
    Returns {"found": bool, "value": str, "confidence_hint": "high"/"medium"/"low"} or None on error.
    This is a best-effort structural parser used only when the primary CV+OCR pipeline could not
    find a mandatory field with sufficient confidence.
    """
    if not ANTHROPIC_SDK_AVAILABLE:
        return None
    try:
        client = anthropic.Anthropic(api_key=api_key)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        field_desc = FALLBACK_FIELD_PROMPTS.get(missing_field_key, missing_field_key)
        prompt = (
            f"You are assisting a Legal Metrology compliance check on a packaged commodity label photo. "
            f"Look ONLY for {field_desc}. "
            f"Respond with STRICT JSON only, no markdown fences, no preamble, in this exact shape: "
            f'{{"found": true or false, "value": "<the exact text you see, or empty string>", '
            f'"confidence_hint": "high" or "medium" or "low"}}. '
            f"If you cannot clearly see this information on the label, set found to false."
        )
        response = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        text_out = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        cleaned = re.sub(r"^```(?:json)?|```$", "", text_out.strip(), flags=re.MULTILINE).strip()
        parsed = json.loads(cleaned)
        return parsed
    except Exception as e:
        return {"found": False, "value": "", "confidence_hint": "low", "error": str(e)}


# ============================================================================================
# CONFIDENCE / NULL PROTOCOL — MERGES PRIMARY + FALLBACK INTO FieldResult LIST
# ============================================================================================

RESCAN_HINTS = {
    "manufacturer": "Manufacturer/packer address block unclear. Upload a clear close-up of the back panel address text.",
    "country_of_origin": "Country of Origin not found. Upload a close-up of the panel stating 'Country of Origin' or 'Made in ___'.",
    "commodity_name": "Product/common name unclear. Upload a straight-on, well-lit photo of the front panel.",
    "net_quantity": "Net Quantity block unclear. Upload a close-up of the weight/volume declaration.",
    "mfg_date": "Manufacturing date stamp unclear. Upload a close-up of the embossed/printed date stamp.",
    "mrp": "MRP block unclear. Please upload a clear close-up of the price/MRP stamp.",
    "usp": "Unit Sale Price not found (required for packs >1kg/1L or multi-unit packs). Upload a close-up of the pricing panel.",
    "consumer_care": "Consumer care details unclear. Upload a close-up of the back panel with contact information.",
}


def evaluate_all_rules(store: EvidenceStore, conf_threshold: float, net_qty_extra: Optional[dict] = None) -> list:
    results = []

    def make_result(key, label, extractor_result, ok_check=lambda r: True, extra_detail=""):
        if extractor_result is None or extractor_result.get("confidence", 0) < conf_threshold:
            return FieldResult(
                field_key=key, label=label, status="NULL",
                value=extractor_result["value"] if extractor_result else "",
                confidence=extractor_result["confidence"] if extractor_result else 0.0,
                detail="Field not found or below the confidence threshold across all uploaded images.",
                rescan_hint=RESCAN_HINTS.get(key, "Please upload a clearer image of this section."),
                rule=RULE_CITATIONS[key],
            )
        compliant = ok_check(extractor_result)
        return FieldResult(
            field_key=key, label=label,
            status="COMPLIANT" if compliant else "NON_COMPLIANT",
            value=extractor_result["value"], confidence=extractor_result["confidence"],
            source=extractor_result.get("source", ""), detail=extra_detail,
            rule=RULE_CITATIONS[key],
        )

    results.append(make_result("manufacturer", "Manufacturer / Packer / Importer", extract_manufacturer(store)))
    results.append(make_result("country_of_origin", "Country of Origin", extract_country_of_origin(store)))
    results.append(make_result("commodity_name", "Commodity / Product Name", extract_commodity_name(store)))

    nq = extract_net_quantity(store)
    if nq is None or nq.get("confidence", 0) < conf_threshold:
        results.append(FieldResult(
            field_key="net_quantity", label="Net Quantity", status="NULL",
            value=nq["value"] if nq else "", confidence=nq["confidence"] if nq else 0.0,
            detail="Net quantity declaration not found or below confidence threshold.",
            rescan_hint=RESCAN_HINTS["net_quantity"], rule=RULE_CITATIONS["net_quantity"],
        ))
    else:
        if nq["violation"]:
            results.append(FieldResult(
                field_key="net_quantity", label="Net Quantity", status="NON_COMPLIANT",
                value=nq["value"], confidence=nq["confidence"], source=nq["source"],
                detail=f"Non-standard unit symbol '{nq['raw_unit']}' used. Statutory symbol is "
                       f"'{nq['expected_symbol']}'. Only standard symbols (g, kg, ml, l, m, cm, N, Pcs) are permitted.",
                rule=RULE_CITATIONS["net_quantity"],
            ))
        else:
            results.append(FieldResult(
                field_key="net_quantity", label="Net Quantity", status="COMPLIANT",
                value=nq["value"], confidence=nq["confidence"], source=nq["source"],
                detail="Statutory unit symbol used correctly.", rule=RULE_CITATIONS["net_quantity"],
            ))

    results.append(make_result("mfg_date", "Month & Year of Manufacture/Packing", extract_mfg_date(store)))

    mrp = extract_mrp(store)
    if mrp is None or mrp.get("confidence", 0) < conf_threshold:
        results.append(FieldResult(
            field_key="mrp", label="Maximum Retail Price (MRP)", status="NULL",
            value=mrp["value"] if mrp else "", confidence=mrp["confidence"] if mrp else 0.0,
            detail="MRP not found or below confidence threshold.",
            rescan_hint=RESCAN_HINTS["mrp"], rule=RULE_CITATIONS["mrp"],
        ))
    else:
        if mrp["has_tax_clause"]:
            results.append(FieldResult(
                field_key="mrp", label="Maximum Retail Price (MRP)", status="COMPLIANT",
                value=mrp["value"], confidence=mrp["confidence"], source=mrp["source"],
                detail="MRP found together with an 'inclusive of all taxes' declaration.",
                rule=RULE_CITATIONS["mrp"],
            ))
        else:
            results.append(FieldResult(
                field_key="mrp", label="Maximum Retail Price (MRP)", status="NON_COMPLIANT",
                value=mrp["value"], confidence=mrp["confidence"], source=mrp["source"],
                detail="MRP found, but no 'inclusive of all taxes' / 'incl. of all taxes' phrase was detected "
                       "nearby, which Rule 6(1)(e) requires.",
                rule=RULE_CITATIONS["mrp"],
            ))

    # USP is conditional — only mandatory if net quantity > 1 kg / 1 L or the pack is a multi-unit pack.
    usp_required = False
    if nq and not nq.get("violation") and nq.get("raw_unit") in ("kg", "l"):
        try:
            usp_required = float(nq["number"]) > 1.0
        except ValueError:
            usp_required = False
    usp = extract_usp(store)
    if usp_required:
        if usp is None or usp.get("confidence", 0) < conf_threshold:
            results.append(FieldResult(
                field_key="usp", label="Unit Sale Price (USP)", status="NULL",
                value=usp["value"] if usp else "", confidence=usp["confidence"] if usp else 0.0,
                detail="Pack exceeds 1 kg/1 L so USP is mandatory, but it was not found.",
                rescan_hint=RESCAN_HINTS["usp"], rule=RULE_CITATIONS["usp"],
            ))
        else:
            results.append(FieldResult(
                field_key="usp", label="Unit Sale Price (USP)", status="COMPLIANT",
                value=usp["value"], confidence=usp["confidence"], source=usp["source"],
                detail="USP declaration found for a multi-unit / >1kg/1L pack.", rule=RULE_CITATIONS["usp"],
            ))
    else:
        results.append(FieldResult(
            field_key="usp", label="Unit Sale Price (USP)", status="COMPLIANT",
            value=usp["value"] if usp else "Not applicable (pack ≤ 1kg/1L)",
            confidence=1.0 if not usp else usp["confidence"],
            detail="USP is not mandatory for this pack size.", rule=RULE_CITATIONS["usp"],
        ))

    cc = extract_consumer_care(store)
    if cc is None or cc.get("confidence", 0) < conf_threshold:
        results.append(FieldResult(
            field_key="consumer_care", label="Consumer Care Details", status="NULL",
            value=cc["value"] if cc else "", confidence=cc["confidence"] if cc else 0.0,
            detail="Consumer care name/address/phone/email not found or below confidence threshold.",
            rescan_hint=RESCAN_HINTS["consumer_care"], rule=RULE_CITATIONS["consumer_care"],
        ))
    else:
        if cc["has_phone"] or cc["has_email"]:
            results.append(FieldResult(
                field_key="consumer_care", label="Consumer Care Details", status="COMPLIANT",
                value=cc["value"], confidence=cc["confidence"], source=cc["source"],
                detail="At least one verifiable contact channel (phone or email) found.", rule=RULE_CITATIONS["consumer_care"],
            ))
        else:
            results.append(FieldResult(
                field_key="consumer_care", label="Consumer Care Details", status="NON_COMPLIANT",
                value=cc["value"], confidence=cc["confidence"], source=cc["source"],
                detail="A 'consumer care' anchor phrase was found but no verifiable phone number or email "
                       "was detected nearby.", rule=RULE_CITATIONS["consumer_care"],
            ))

    return results


# ============================================================================================
# REPORT GENERATION
# ============================================================================================

STATUS_ICON = {"COMPLIANT": "🟢", "NON_COMPLIANT": "🔴", "NULL": "🟡"}


def build_report_text(product_name: str, results: list, pdp_result: FieldResult) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("METRASIGHT — DIGITAL LEGAL METROLOGY INSPECTION REPORT")
    lines.append("Legal Metrology (Packaged Commodities) Rules, 2011")
    lines.append("=" * 78)
    lines.append(f"Product: {product_name or 'Unnamed Product'}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("-" * 78)
    all_results = results + [pdp_result]
    n_compliant = sum(1 for r in all_results if r.status == "COMPLIANT")
    n_noncompliant = sum(1 for r in all_results if r.status == "NON_COMPLIANT")
    n_null = sum(1 for r in all_results if r.status == "NULL")
    lines.append(f"Summary: {n_compliant} Compliant | {n_noncompliant} Non-Compliant | {n_null} Null/Re-scan needed")
    lines.append("-" * 78)
    for r in all_results:
        lines.append("")
        lines.append(f"{STATUS_ICON[r.status]} [{r.status}] {r.label}")
        lines.append(f"   Rule: {r.rule}")
        lines.append(f"   Detected Value: {r.value or '(none)'}")
        lines.append(f"   Confidence: {r.confidence:.0%}")
        if r.source:
            lines.append(f"   Source: {r.source}")
        if r.detail:
            lines.append(f"   Detail: {r.detail}")
        if r.status == "NULL" and r.rescan_hint:
            lines.append(f"   ⚠ EVIDENCE PLANNER ALERT: {r.rescan_hint}")
    lines.append("")
    lines.append("=" * 78)
    lines.append("This report is generated by an automated prototype tool (SIH26034 MetraSight) and")
    lines.append("does not constitute a formal legal determination. Fields marked NULL require manual")
    lines.append("re-inspection before any enforcement action.")
    lines.append("=" * 78)
    return "\n".join(lines)


def build_report_csv(product_name: str, results: list, pdp_result: FieldResult) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Product", product_name or "Unnamed Product"])
    writer.writerow(["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow([])
    writer.writerow(["Rule Citation", "Field", "Status", "Detected Value", "Confidence", "Source", "Detail", "Re-scan Hint"])
    for r in results + [pdp_result]:
        writer.writerow([r.rule, r.label, r.status, r.value, f"{r.confidence:.2f}", r.source, r.detail, r.rescan_hint])
    return output.getvalue()


# ============================================================================================
# STREAMLIT APP
# ============================================================================================

st.set_page_config(page_title="MetraSight | AI Legal-Metrology Inspector", page_icon="⚖️", layout="wide")

CUSTOM_CSS = """
<style>
.main { background-color: #0e1117; }
.metrasight-header {
    background: linear-gradient(90deg, #0b3d91 0%, #1b5fae 60%, #12805c 100%);
    padding: 1.4rem 1.6rem; border-radius: 14px; margin-bottom: 1.2rem;
}
.metrasight-header h1 { color: white; margin: 0; font-size: 1.9rem; }
.metrasight-header p { color: #dbe9ff; margin: 0.3rem 0 0 0; font-size: 0.95rem; }
.compliance-card {
    border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;
    border: 1px solid rgba(255,255,255,0.08);
}
.card-compliant { background-color: rgba(19, 128, 92, 0.12); border-left: 5px solid #13805c; }
.card-noncompliant { background-color: rgba(200, 40, 40, 0.10); border-left: 5px solid #c82828; }
.card-null { background-color: rgba(210, 160, 20, 0.12); border-left: 5px solid #d2a014; }
.card-title { font-weight: 700; font-size: 1.02rem; margin-bottom: 0.25rem; }
.card-rule { font-size: 0.78rem; opacity: 0.75; margin-bottom: 0.4rem; }
.card-value { font-size: 0.92rem; margin-bottom: 0.2rem; }
.alert-box {
    background-color: rgba(210, 160, 20, 0.18); border: 1px dashed #d2a014;
    border-radius: 8px; padding: 0.6rem 0.9rem; margin-top: 0.4rem; font-size: 0.85rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="metrasight-header">
        <h1>⚖️ MetraSight — AI Legal-Metrology Inspector</h1>
        <p>SIH26034 · Ministry of Consumer Affairs, Food & Public Distribution ·
        Legal Metrology (Packaged Commodities) Rules, 2011 compliance engine</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not EASYOCR_AVAILABLE:
    st.error(
        "EasyOCR is not installed in this environment. Install the dependencies in "
        "`requirements.txt` (this will download PyTorch, so first install may take a few minutes)."
    )

# ---------------------------- SESSION STATE ----------------------------
if "evidence_store" not in st.session_state:
    st.session_state.evidence_store = EvidenceStore()
if "processed_images" not in st.session_state:
    st.session_state.processed_images = {}   # filename -> PIL image with boxes
if "raw_ocr_by_image" not in st.session_state:
    st.session_state.raw_ocr_by_image = {}
if "rule_results" not in st.session_state:
    st.session_state.rule_results = []
if "pdp_result" not in st.session_state:
    st.session_state.pdp_result = None

# ---------------------------- SIDEBAR ----------------------------
with st.sidebar:
    st.header("⚙️ Inspection Settings")
    product_name = st.text_input("Product Name (for the report header)", value="")

    st.subheader("Pre-processing")
    use_grayscale = st.checkbox("Grayscale conversion", value=True)
    use_clahe = st.checkbox("CLAHE contrast stretching", value=True)
    use_denoise = st.checkbox("Denoise (reduces glare speckle)", value=True)
    use_adaptive_threshold = st.checkbox("Adaptive thresholding", value=False)

    st.subheader("Confidence & PDP")
    conf_threshold = st.slider("Minimum field confidence to accept (%)", 0, 100, 50) / 100.0
    pdp_area_cm2 = st.slider("Principal Display Panel (PDP) area (cm²)", 1.0, 3000.0, 100.0)
    px_per_mm = st.number_input(
        "Calibration: pixels per millimetre in the uploaded photo",
        min_value=0.0, value=10.0, step=0.5,
        help="Needed to convert detected text height in pixels to real-world mm for the Rule 7 numeral-height check. "
             "Estimate by measuring a known-size reference object in the photo (e.g. a 1 cm grid or ruler).",
    )

    st.subheader("Fallback: Claude Vision Parser")
    enable_fallback = st.checkbox("Enable Claude Vision fallback for NULL fields", value=False)
    claude_api_key = ""
    claude_model = DEFAULT_CLAUDE_MODEL
    if enable_fallback:
        if not ANTHROPIC_SDK_AVAILABLE:
            st.warning("The `anthropic` package is not installed. Add it to requirements.txt to use this feature.")
        claude_api_key = st.text_input("Your Anthropic API key", type="password",
                                        help="Your key is used only for this session and is never stored.")
        claude_model = st.text_input(
            "Claude model string", value=DEFAULT_CLAUDE_MODEL,
            help="Check docs.claude.com for the current list of available model strings for your account.",
        )

    st.markdown("---")
    st.caption(
        "⚠️ The Rule 7 numeral-height table used in this prototype is an indicative approximation. "
        "Verify against the official Second Schedule before relying on this tool for real inspections."
    )

# ---------------------------- TABS ----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Upload & Pre-process",
    "2️⃣ OCR Evidence Store",
    "3️⃣ Compliance Report",
    "4️⃣ Download Audit Report",
])

# ---------------------------- TAB 1: UPLOAD & PREPROCESS ----------------------------
with tab1:
    st.subheader("Multi-Image Upload")
    st.write("Upload every available panel of the SAME product (front, back, top/bottom, close-ups). "
             "MetraSight pools evidence across all images before evaluating compliance.")
    uploaded_files = st.file_uploader(
        "Upload product images", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True
    )

    run_button = st.button("🔍 Run Pre-processing + OCR on all images", type="primary", disabled=not uploaded_files)

    if run_button and uploaded_files:
        reader = get_ocr_reader()
        if reader is None:
            st.error("EasyOCR could not be loaded. Check requirements.txt / installation logs.")
        else:
            store = EvidenceStore()
            processed_images = {}
            raw_ocr_by_image = {}
            progress = st.progress(0.0, text="Starting...")
            for i, uf in enumerate(uploaded_files):
                progress.progress((i) / len(uploaded_files), text=f"Processing {uf.name}...")
                pil_img = Image.open(uf)
                pre_img = preprocess_pipeline(
                    pil_img, use_grayscale, use_clahe, use_denoise, use_adaptive_threshold
                )
                ocr_rows = run_ocr(pre_img, reader)
                for row in ocr_rows:
                    row["source"] = uf.name
                    store.raw_tokens.append(row)
                store.full_text += "\n" + "\n".join(r["text"] for r in ocr_rows)
                raw_ocr_by_image[uf.name] = ocr_rows

                boxed_display = draw_bounding_boxes(pil_img.convert("RGB"), ocr_rows, conf_threshold)
                processed_images[uf.name] = {"preprocessed": pre_img, "boxed": boxed_display}

            progress.progress(1.0, text="Done.")
            st.session_state.evidence_store = store
            st.session_state.processed_images = processed_images
            st.session_state.raw_ocr_by_image = raw_ocr_by_image
            st.success(
                f"Processed {len(uploaded_files)} image(s) — {len(store.raw_tokens)} text regions detected. "
                f"Go to Tab 2 to review evidence, or Tab 3 for the compliance report."
            )

    if st.session_state.processed_images:
        st.markdown("---")
        st.subheader("Preview: Pre-processed Images & Detected Text Regions")
        cols = st.columns(2)
        for idx, (fname, imgs) in enumerate(st.session_state.processed_images.items()):
            with cols[idx % 2]:
                st.markdown(f"**{fname}**")
                st.image(imgs["boxed"], caption="Bounding boxes (green = high confidence, red = low)", use_container_width=True)
                with st.expander("View pre-processed (cleaned) image"):
                    st.image(imgs["preprocessed"], use_container_width=True)

# ---------------------------- TAB 2: OCR EVIDENCE STORE ----------------------------
with tab2:
    st.subheader("Aggregated Raw OCR Output")
    store: EvidenceStore = st.session_state.evidence_store
    if not store.raw_tokens:
        st.info("No OCR evidence yet. Upload images and run pre-processing + OCR in Tab 1.")
    else:
        st.code(store.full_text.strip() or "(empty)", language="text")

        st.subheader("Per-token Detail (all uploaded images combined)")
        df = pd.DataFrame([
            {"Text": t["text"], "Confidence": round(t["confidence"], 2), "Height (px)": round(t["height_px"], 1),
             "Source Image": t["source"]}
            for t in store.raw_tokens
        ])
        st.dataframe(df, use_container_width=True, height=320)

        st.subheader("Fuzzy Token Matches (unit / anchor phrase detection)")
        fuzzy_hits = []
        for kw in ["consumer care", "customer care", "manufactured by", "marketed by", "packed by",
                   "country of origin", "made in", "mrp", "inclusive of all taxes"]:
            match = fuzzy_contains(store.full_text, [kw], threshold=0.72)
            if match:
                fuzzy_hits.append({"Anchor Phrase": kw, "Matched In Text": "Yes"})
        unit_hits = []
        for tok in store.raw_tokens:
            for bad_unit, correct in NON_STANDARD_UNIT_MAP.items():
                if re.search(rf"\b{re.escape(bad_unit)}\b", tok["text"], re.IGNORECASE):
                    unit_hits.append({"Non-standard Unit Found": bad_unit, "Should Be": correct, "Source": tok["source"]})
        if fuzzy_hits:
            st.dataframe(pd.DataFrame(fuzzy_hits), use_container_width=True)
        if unit_hits:
            st.warning("Non-standard unit tokens detected (Rule 6(1)(c) risk):")
            st.dataframe(pd.DataFrame(unit_hits), use_container_width=True)
        if not fuzzy_hits and not unit_hits:
            st.caption("No fuzzy anchor phrases or non-standard units detected.")

# ---------------------------- TAB 3: COMPLIANCE REPORT ----------------------------
with tab3:
    st.subheader("Structured Compliance Verdicts")
    store: EvidenceStore = st.session_state.evidence_store
    if not store.raw_tokens:
        st.info("No evidence to evaluate yet. Upload images and run OCR in Tab 1.")
    else:
        if st.button("🧮 Evaluate Compliance Rules", type="primary"):
            results = evaluate_all_rules(store, conf_threshold)
            pdp_result = check_pdp_numerals(store, pdp_area_cm2, px_per_mm)

            # ---- OPTIONAL FALLBACK: Claude Vision for any NULL mandatory field ----
            if enable_fallback and claude_api_key and ANTHROPIC_SDK_AVAILABLE:
                null_fields = [r for r in results if r.status == "NULL"]
                if null_fields and st.session_state.processed_images:
                    with st.spinner(f"Running Claude Vision fallback for {len(null_fields)} field(s)..."):
                        for r in null_fields:
                            for fname in st.session_state.processed_images:
                                try:
                                    buf = io.BytesIO()
                                    st.session_state.processed_images[fname]["preprocessed"].convert("RGB").save(buf, format="JPEG")
                                    fb = call_claude_vision_fallback(claude_api_key, claude_model, buf.getvalue(), r.field_key)
                                except Exception as e:
                                    fb = {"found": False, "error": str(e)}
                                if fb and fb.get("found"):
                                    r.value = fb.get("value", r.value)
                                    hint = fb.get("confidence_hint", "medium")
                                    r.confidence = {"high": 0.85, "medium": 0.65, "low": 0.45}.get(hint, 0.5)
                                    r.source = f"Claude Vision / {fname}"
                                    if r.field_key == "net_quantity":
                                        # Vision fallback only recovers the raw text; it does not re-run the
                                        # statutory-unit RegEx check, so flag this for manual unit verification.
                                        r.status = "NON_COMPLIANT"
                                        r.detail = (f"Recovered via Claude Vision fallback (image: {fname}), but the "
                                                    f"statutory-unit-symbol check could not be re-run automatically — "
                                                    f"please manually verify the unit symbol is one of g, kg, ml, l, m, cm, N, Pcs.")
                                    else:
                                        r.status = "COMPLIANT"
                                        r.detail = f"Recovered via Claude Vision fallback parser (image: {fname})."
                                    break

            st.session_state.rule_results = results
            st.session_state.pdp_result = pdp_result

        if st.session_state.rule_results:
            results = st.session_state.rule_results
            pdp_result = st.session_state.pdp_result
            all_results = results + ([pdp_result] if pdp_result else [])

            n_compliant = sum(1 for r in all_results if r.status == "COMPLIANT")
            n_noncompliant = sum(1 for r in all_results if r.status == "NON_COMPLIANT")
            n_null = sum(1 for r in all_results if r.status == "NULL")
            c1, c2, c3 = st.columns(3)
            c1.metric("🟢 Compliant", n_compliant)
            c2.metric("🔴 Non-Compliant", n_noncompliant)
            c3.metric("🟡 Null / Re-scan Needed", n_null)

            st.markdown("---")
            for r in all_results:
                css_class = {"COMPLIANT": "card-compliant", "NON_COMPLIANT": "card-noncompliant", "NULL": "card-null"}[r.status]
                icon = STATUS_ICON[r.status]
                html = f"""
                <div class="compliance-card {css_class}">
                    <div class="card-title">{icon} {r.label} — {r.status.replace('_',' ')}</div>
                    <div class="card-rule">{r.rule}</div>
                    <div class="card-value"><b>Detected:</b> {r.value or '(none)'} &nbsp;|&nbsp; <b>Confidence:</b> {r.confidence:.0%}
                    {'&nbsp;|&nbsp; <b>Source:</b> ' + r.source if r.source else ''}</div>
                    {'<div class="card-value">' + r.detail + '</div>' if r.detail else ''}
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)
                if r.status == "NULL" and r.rescan_hint:
                    st.markdown(f'<div class="alert-box">📸 <b>Evidence Planner Alert:</b> {r.rescan_hint}</div>',
                                unsafe_allow_html=True)
        else:
            st.caption("Click 'Evaluate Compliance Rules' to generate verdicts from the pooled evidence.")

# ---------------------------- TAB 4: DOWNLOAD AUDIT REPORT ----------------------------
with tab4:
    st.subheader("Downloadable Official Inspection Report")
    if not st.session_state.rule_results:
        st.info("Run the compliance evaluation in Tab 3 first.")
    else:
        results = st.session_state.rule_results
        pdp_result = st.session_state.pdp_result
        report_txt = build_report_text(product_name, results, pdp_result)
        report_csv = build_report_csv(product_name, results, pdp_result)

        st.text_area("Report Preview", report_txt, height=400)

        colA, colB = st.columns(2)
        with colA:
            st.download_button(
                "⬇️ Download Report (.txt)", data=report_txt,
                file_name=f"metrasight_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )
        with colB:
            st.download_button(
                "⬇️ Download Report (.csv)", data=report_csv,
                file_name=f"metrasight_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

st.markdown("---")
st.caption(
    "MetraSight prototype for SIH26034. Automated verdicts are decision-support only and must be "
    "confirmed by a human inspector before any regulatory action, especially any field the tool "
    "returns as NULL / Re-scan needed."
)

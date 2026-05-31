"""
regex_extractor.py — Deterministic regex-based extraction of PA policy parameters.

Fast first pass that parses structured PDF sections (Required Medical Information,
Age Restrictions, Prescriber Restrictions, Coverage Duration, Renewal Criteria)
without any LLM calls. Returns snake_case keys matching the pipeline's internal
format.

Parameters it can't extract are returned as None (not "NA") so the caller can
distinguish "not found by regex" from "explicitly absent in the document".
"""
import re
from typing import Any, Optional

import fitz

# ── Drug aliases for section detection ───────────────────────────────────────
DRUG_ALIASES = {
    "ACITRETIN": ["ACITRETIN", "SORIATANE"],
    "SKYRIZI": ["SKYRIZI", "RISANKIZUMAB"],
    "TREMFYA": ["TREMFYA", "GUSELKUMAB"],
    "STELARA": ["STELARA", "USTEKINUMAB"],
    "ILUMYA": ["ILUMYA", "TILDRAKIZUMAB"],
    "COSENTYX": ["COSENTYX", "SECUKINUMAB"],
    "TALTZ": ["TALTZ", "IXEKIZUMAB"],
    "HUMIRA": ["HUMIRA", "ADALIMUMAB"],
    "ENBREL": ["ENBREL", "ETANERCEPT"],
    "CIMZIA": ["CIMZIA", "CERTOLIZUMAB"],
    "SIMPONI": ["SIMPONI", "GOLIMUMAB"],
    "OTEZLA": ["OTEZLA", "APREMILAST"],
    "SOTYKTU": ["SOTYKTU", "DEUCRAVACITINIB"],
    "RINVOQ": ["RINVOQ", "UPADACITINIB"],
    "XELJANZ": ["XELJANZ", "TOFACITINIB"],
    "BIMZELX": ["BIMZELX", "BIMEKIZUMAB"],
    "SILIQ": ["SILIQ", "BRODALUMAB"],
}

SPECIALIST_MAP = {
    "dermatologist": "Dermatologist",
    "rheumatologist": "Rheumatologist",
    "gastroenterologist": "Gastroenterologist",
    "immunologist": "Immunologist",
    "infectious disease specialist": "Infectious Disease Specialist",
}

GENERIC_TERMS = [
    "topical corticosteroids", "corticosteroids", "vitamin d analogs",
    "vitamin d analogues", "tazorac", "tazarotene", "topical tacrolimus",
    "tacrolimus", "elidel", "pimecrolimus", "methotrexate", "cyclosporine",
    "acitretin", "conventional therapies", "topical therapies", "non-biologic",
]


# ── Text utilities ───────────────────────────────────────────────────────────

def _normalize_space(value: str) -> str:
    value = value.replace("\x00", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n\s+", "\n", value)
    value = re.sub(r"\s+\n", "\n", value)
    return value.strip()


def _compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _casefold_terms(drug_name: str) -> list[str]:
    terms = [drug_name]
    terms.extend(DRUG_ALIASES.get(drug_name.upper(), []))
    seen = set()
    deduped = []
    for t in terms:
        key = t.casefold()
        if key not in seen:
            deduped.append(t)
            seen.add(key)
    return deduped


# ── PDF section extraction ───────────────────────────────────────────────────

def _page_mentions_target(page_text: str, terms: list[str]) -> bool:
    low = page_text.casefold()
    return any(t.casefold() in low for t in terms)


def _page_looks_like_section_start(page_text: str) -> bool:
    first = _compact(page_text[:900]).casefold()
    return ("products affected" in first
            and ("prior authorization criteria" in first
                 or "criteria details" in first))


def _find_start_page(pages: list[str], drug_name: str) -> Optional[int]:
    terms = _casefold_terms(drug_name)
    for i, page in enumerate(pages):
        if not _page_mentions_target(page, terms):
            continue
        first = page[:1200].casefold()
        if ("products affected" in first
                or "drug names" in first
                or "prior authorization group" in first):
            return i
    return None


def _extract_section(pages: list[str], drug_name: str) -> Optional[str]:
    terms = _casefold_terms(drug_name)
    start = _find_start_page(pages, drug_name)
    if start is None:
        return None

    section_pages = [pages[start]]
    for i in range(start + 1, len(pages)):
        page = pages[i]
        if (_page_looks_like_section_start(page)
                and not _page_mentions_target(page, terms)):
            break
        section_pages.append(page)

    return _normalize_space("\n".join(section_pages))


# ── Block extractors ─────────────────────────────────────────────────────────

def _extract_between(section: str, start_pat: str,
                     end_pats: list[str]) -> Optional[str]:
    end = "|".join(end_pats)
    m = re.search(rf"(?is){start_pat}\s*(.*?)(?={end}|\Z)", section)
    if not m:
        return None
    val = _normalize_space(m.group(1))
    val = re.sub(r"^(Information|Restrictions|Duration|Criteria)\s+",
                 "", val, flags=re.IGNORECASE)
    val = _normalize_space(val)
    if val in ("", "-", "—"):
        return None
    return val


_END_LABELS = [
    r"Age\s+Restrictions",
    r"Prescriber\s+Restrictions",
    r"Coverage\s+Duration",
    r"Renewal\s+Criteria",
    r"Effective\s+Date",
    r"P&T\s+Approval\s+Date",
    r"P&T\s+Revision\s+Date",
]


def _get_required_medical_info(section: str) -> Optional[str]:
    return _extract_between(section, r"Required\s+Medical\s+Information", _END_LABELS)


def _get_age_block(section: str) -> Optional[str]:
    return _extract_between(section, r"Age\s+Restrictions", _END_LABELS[1:])


def _get_prescriber_block(section: str) -> Optional[str]:
    return _extract_between(section, r"Prescriber\s+Restrictions", _END_LABELS[2:])


def _get_coverage_block(section: str) -> Optional[str]:
    return _extract_between(section, r"Coverage\s+Duration", _END_LABELS[3:])


def _get_renewal_block(section: str) -> Optional[str]:
    return _extract_between(section, r"Renewal\s+Criteria", _END_LABELS[4:])


# ── Parameter extractors ─────────────────────────────────────────────────────

def _extract_age(section: str) -> Optional[str]:
    block = _get_age_block(section)
    if not block:
        return None
    text = _compact(block).casefold()

    for pat in [
        r"(\d{1,2})\s*years?\s*of\s*age\s*or\s*older",
        r"at\s*least\s*(\d{1,2})\s*years?\s*of\s*age",
        r"(\d{1,2})\s*years?\s*and\s*older",
        r"aged?\s*(\d{1,2})\s*or\s*older",
        r"must\s*be\s*(?:at\s*least\s*)?(\d{1,2})\s*years?",
    ]:
        m = re.search(pat, text)
        if m:
            return f">={m.group(1)}"

    # Pregnancy monitoring is safety criteria, not age eligibility
    if any(w in text for w in ("childbearing", "able to bear children", "pregnancy")):
        return None

    return _compact(block)


def _extract_step_therapy_text(info: Optional[str]) -> Optional[str]:
    if not info:
        return None
    text = _compact(info).replace("• ", "")
    return re.sub(r"\s+", " ", text).strip() or None


def _extract_brand_steps(info: Optional[str]) -> Optional[int]:
    if not info:
        return None
    low = info.casefold()

    for pat in [
        r"trial\s+and\s+failure.*?\b(two|2)\b.*?(biologic|tnf|humira|enbrel|stelara|skyrizi|tremfya|cimzia)",
        r"trial\s+and\s+failure.*?\b(one|1)\b.*?(biologic|tnf|humira|enbrel|stelara|skyrizi|tremfya|cimzia)",
    ]:
        m = re.search(pat, low)
        if m:
            w = m.group(1)
            return 2 if w == "two" else (1 if w == "one" else int(w))

    if "biologic as first-line" in low:
        return 0
    return None


def _extract_generic_steps(info: Optional[str]) -> Optional[int]:
    if not info:
        return None
    low = info.casefold()

    if re.search(r"at\s+least\s+(?:2|two)\s+conventional\s+therap", low):
        return 2
    if re.search(r"\b(?:2|two)\s+(?:topical|generic|non-biologic|conventional)", low):
        return 2
    if any(t in low for t in GENERIC_TERMS):
        return 1
    return None


def _extract_phototherapy(info: Optional[str]) -> Optional[str]:
    if not info:
        return None
    low = info.casefold()
    if not any(t in low for t in ("phototherapy", "puva", "uvb")):
        return None
    if "one of the following" in low or "conventional therapies" in low or "or " in low:
        return "No"  # optional, not mandatory
    return "Yes"


def _extract_tb(section: str) -> Optional[str]:
    low = section.casefold()
    if re.search(r"\bnegative\s+tuberculin\b|\btb\s+test\b|\btuberculosis\b|\blatent\s+tb\b", low):
        return "Y"
    return None


def _extract_quantity_limits(section: str) -> Optional[str]:
    low = section.casefold()
    if "quantity limit" in low or "quantity limits" in low:
        return "Yes"
    return None


def _extract_specialist(section: str) -> Optional[str]:
    block = _get_prescriber_block(section)
    if not block:
        return None
    low = block.casefold()
    found = []
    for needle, label in SPECIALIST_MAP.items():
        if needle in low and label not in found:
            found.append(label)
    return ", ".join(found) if found else None


def _extract_initial_duration(section: str) -> Optional[int]:
    cov = _get_coverage_block(section) or section
    m = re.search(r"(?i)\binitial\s*:\s*(\d+)\s*months?", cov)
    if m:
        return int(m.group(1))
    m = re.search(r"(?i)\binitial\s*:\s*(\d+)\s*weeks?", cov)
    if m:
        return round(int(m.group(1)) / 4)
    return None


def _extract_reauth_duration(section: str) -> Optional[int]:
    cov = _get_coverage_block(section) or section
    for pat in [
        r"(?i)\brenewal\s*:\s*(\d+)\s*months?",
        r"(?i)\breauthorization\s*:\s*(\d+)\s*months?",
    ]:
        m = re.search(pat, cov)
        if m:
            return int(m.group(1))
    m = re.search(r"(?i)\brenewal\s*:\s*(\d+)\s*weeks?", cov)
    if m:
        return round(int(m.group(1)) / 4)
    return None


def _extract_reauth_required(section: str) -> Optional[str]:
    if _extract_reauth_duration(section) is not None:
        return "Yes"
    renewal = _get_renewal_block(section)
    if renewal and renewal.strip() not in ("-", "—"):
        return "Yes"
    return None


def _extract_reauth_requirements(section: str) -> Optional[str]:
    renewal = _get_renewal_block(section)
    if not renewal:
        return None
    text = _compact(renewal)
    text = re.sub(r"(?i)^Renewal Criteria:\s*", "", text)
    return text.strip() or None


# ── Public API ───────────────────────────────────────────────────────────────

def regex_extract(pdf_path: str, brand: str) -> Optional[dict]:
    """Try deterministic regex extraction from a PDF.

    Only works on structured PA-format PDFs (with "Products Affected" /
    "Prior Authorization Criteria" sections). Returns None for other formats
    rather than guessing — the LLM handles unstructured documents.

    Returns a dict with snake_case keys. Values are None for parameters
    that couldn't be extracted from the structured section.
    """
    doc = fitz.open(pdf_path)
    try:
        pages = [page.get_text("text") for page in doc]
    finally:
        doc.close()

    section = _extract_section(pages, brand)
    if not section:
        return None

    # Verify this is actually a structured PA section (not a false match)
    low = section[:500].lower()
    structured_markers = ["products affected", "prior authorization criteria",
                          "criteria details", "required medical information"]
    if sum(1 for m in structured_markers if m in low) < 2:
        # Not enough structural markers — don't trust regex on this format
        return None

    info = _get_required_medical_info(section)

    result = {
        "age": _extract_age(section),
        "step_therapy_requirements": _extract_step_therapy_text(info),
        "steps_through_brands": _extract_brand_steps(info),
        "steps_through_generic": _extract_generic_steps(info),
        "step_through_phototherapy": _extract_phototherapy(info),
        "tb_test_required": _extract_tb(section),
        "quantity_limits": _extract_quantity_limits(section),
        "specialist_types": _extract_specialist(section),
        "initial_auth_duration": _extract_initial_duration(section),
        "reauth_duration": _extract_reauth_duration(section),
        "reauth_required": _extract_reauth_required(section),
        "reauth_requirements": _extract_reauth_requirements(section),
    }

    extracted = sum(1 for v in result.values() if v is not None)
    print(f"    [regex] structured PA format, extracted {extracted}/12 params "
          f"({len(section):,} chars)")

    return result


def merge_regex_llm(regex_result: dict, llm_result: dict) -> dict:
    """Merge regex and LLM results.

    Regex values take priority for structured fields (durations, yes/no flags,
    counts) since regex is deterministic and precise on structured PA formats.
    For free-text fields (step_therapy_requirements, reauth_requirements,
    reasoning), prefer the LLM result as it captures nuance better.

    Both dicts use snake_case keys. None values in regex_result are filled
    from llm_result.
    """
    # Fields where regex is more reliable than LLM (structured/numeric)
    REGEX_PRIORITY = {
        "age", "steps_through_brands", "steps_through_generic",
        "step_through_phototherapy", "tb_test_required", "quantity_limits",
        "specialist_types", "initial_auth_duration", "reauth_duration",
        "reauth_required",
    }

    merged = {}
    for key in llm_result:
        regex_val = regex_result.get(key)
        llm_val = llm_result[key]

        if regex_val is not None and key in REGEX_PRIORITY:
            merged[key] = regex_val
        elif regex_val is not None and llm_val in (None, "", "NA", "N/A"):
            # LLM missed it, regex found it — use regex
            merged[key] = regex_val
        else:
            merged[key] = llm_val

    # Include any keys only in regex_result
    for key in regex_result:
        if key not in merged and regex_result[key] is not None:
            merged[key] = regex_result[key]
    return merged

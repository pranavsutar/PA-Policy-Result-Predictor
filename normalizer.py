"""
normalizer.py — Deterministic post-LLM normalization.

Standardizes raw LLM extraction output into consistent formats
before scoring. Runs after every extraction, before caching.
"""
import re

# ── Word-to-number mapping ──────────────────────────────────────────────
_WORD_TO_NUM = {
    "zero": 0, "none": 0, "no": 0,
    "one": 1, "a single": 1,
    "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# Values treated as NA across all fields
_NA_VARIANTS = {
    "na", "n/a", "not applicable", "not mentioned", "not specified",
    "not stated", "none", "none mentioned", "none specified",
    "none required", "not available", "not documented", "unknown",
    "not addressed", "not indicated", "not listed", "",
}


def _is_na(val: str) -> bool:
    """Check if a value is effectively NA."""
    return val.strip().lower() in _NA_VARIANTS


def _normalize_yes_no(val: str) -> str:
    """Normalize a value to Yes / No / NA.

    Handles: "Y", "Required", "TB screening required", "true", etc.
    """
    if _is_na(val):
        return "NA"

    lower = val.strip().lower()

    # Explicit No
    if lower in ("no", "false", "n", "not required", "no requirement",
                 "no restriction", "not needed"):
        return "No"

    # Explicit Yes
    if lower in ("yes", "true", "y", "required", "mandatory"):
        return "Yes"

    # Phrases that imply Yes
    yes_signals = [
        "required", "must", "needed", "necessary", "screening",
        "is required", "are required", "shall", "should be",
    ]
    if any(sig in lower for sig in yes_signals):
        return "Yes"

    # Phrases that imply No
    no_signals = [
        "not required", "no requirement", "not needed", "not necessary",
        "not mandatory", "optional",
    ]
    if any(sig in lower for sig in no_signals):
        return "No"

    # If it's a short value we can't classify, return as-is
    if len(val.strip()) <= 5:
        return val.strip()

    # Long descriptive text — likely Yes with details
    return "Yes"


def _parse_number(val: str) -> int | None:
    """Extract a number from a string. Handles words and digits."""
    val = val.strip().lower()

    # Direct integer
    m = re.match(r'^(\d+)', val)
    if m:
        return int(m.group(1))

    # Word-to-number
    for word, num in _WORD_TO_NUM.items():
        if word in val:
            return num

    return None


def _normalize_step_count(val: str) -> str:
    """Normalize step count to an integer string or NA."""
    if _is_na(val):
        return "NA"

    n = _parse_number(val)
    if n is not None:
        return str(n)

    # "NA" if we can't parse
    return "NA"


def _normalize_duration(val: str) -> str:
    """Normalize authorization/reauth duration to integer months or NA.

    Handles: "12 months", "1 year", "365 days", "6", "one year",
    "52 weeks", "indefinite", etc.
    """
    if _is_na(val):
        return "NA"

    lower = val.strip().lower()

    # "Indefinite" / "ongoing" / "lifetime"
    if any(w in lower for w in ("indefinite", "ongoing", "lifetime",
                                 "no limit", "unlimited")):
        return "NA"  # No fixed duration = effectively no restriction

    # Try to extract a number first
    n = _parse_number(val)

    if n is not None:
        # Determine unit
        if "year" in lower:
            return str(n * 12)
        elif "week" in lower:
            return str(max(1, n // 4))  # weeks to months
        elif "day" in lower:
            return str(max(1, n // 30))  # days to months
        else:
            # Bare number — assume months if <= 36, otherwise days
            if n <= 36:
                return str(n)
            elif n <= 400:
                return str(max(1, n // 30))  # likely days
            else:
                return str(n)  # unclear, keep as-is

    # "1 year" as words
    for word, num in _WORD_TO_NUM.items():
        if word in lower:
            if "year" in lower:
                return str(num * 12)
            return str(num)

    return "NA"


def _normalize_age(val: str) -> str:
    """Normalize age restriction to a standard format.

    Output: ">=N", "No", or "NA"
    Handles: "18 years or older", "Adults", ">=6 years", "Pediatric (>=4)",
    "No age restriction", "18+", etc.
    """
    if _is_na(val):
        return "NA"

    lower = val.strip().lower()

    # No restriction
    if any(w in lower for w in ("no age", "no restriction", "all ages",
                                 "any age", "no limit")):
        return "No"

    # Extract numeric age
    # Patterns: ">=18", "≥18", "18+", "18 years", "age 18", "at least 18"
    m = re.search(r'(?:>=|≥|>)\s*(\d+)', val)
    if m:
        return f">={m.group(1)}"

    m = re.search(r'(\d+)\s*\+', val)
    if m:
        return f">={m.group(1)}"

    m = re.search(r'(\d+)\s*(?:years?\s*(?:of\s*age|or\s*older|and\s*older))',
                  lower)
    if m:
        return f">={m.group(1)}"

    m = re.search(r'(?:at\s*least|minimum|min)\s*(\d+)', lower)
    if m:
        return f">={m.group(1)}"

    m = re.search(r'age\s*(?:of\s*)?(\d+)', lower)
    if m:
        return f">={m.group(1)}"

    # "Adults" = >=18, "Pediatric" = >=4 or >=6 (ambiguous, keep as-is)
    if "adult" in lower and "pediatric" not in lower:
        return ">=18"

    # Bare number
    m = re.match(r'^(\d+)$', val.strip())
    if m:
        return f">={m.group(1)}"

    # Can't parse — return cleaned original
    return val.strip()


def _normalize_specialist(val: str) -> str:
    """Normalize specialist types to title case, comma-separated, or NA.

    Handles: "dermatologist", "Dermatologist or Rheumatologist",
    "prescribed by a specialist", "Any provider", etc.
    """
    if _is_na(val):
        return "NA"

    lower = val.strip().lower()

    # No specialist requirement
    if any(w in lower for w in ("any provider", "any prescriber",
                                 "no specialist", "no requirement",
                                 "not required", "any physician")):
        return "NA"

    # Known specialist types
    specialists = []
    specialist_map = {
        "dermatologist": "Dermatologist",
        "rheumatologist": "Rheumatologist",
        "gastroenterologist": "Gastroenterologist",
        "immunologist": "Immunologist",
        "allergist": "Allergist",
        "oncologist": "Oncologist",
    }
    for key, title in specialist_map.items():
        if key in lower:
            specialists.append(title)

    if specialists:
        return ", ".join(sorted(set(specialists)))

    # Generic "specialist" mention
    if "specialist" in lower or "board-certified" in lower:
        return "Specialist"

    # Short unrecognized value
    if len(val.strip()) <= 30:
        return val.strip().title()

    return val.strip()


def _normalize_step_therapy(val: str) -> str:
    """Normalize step therapy requirements field.

    Short answers → Yes/No/NA.
    Long descriptive text (>20 chars) → preserved as-is (contains details).
    """
    if _is_na(val):
        return "NA"

    lower = val.strip().lower()

    # Explicit No
    if lower in ("no", "no step therapy", "no step therapy required",
                 "not required", "no requirements", "false"):
        return "No"

    # Explicit Yes (short)
    if lower in ("yes", "true", "required", "step therapy required"):
        return "Yes"

    # Long descriptive text — preserve it (contains the actual requirements)
    if len(val.strip()) > 20:
        return val.strip()

    # Short but unclear
    if any(w in lower for w in ("must", "require", "fail", "tried",
                                 "prior", "step")):
        return "Yes"

    return val.strip()


def _normalize_quantity_limits(val: str) -> str:
    """Normalize quantity limits to Yes / No / NA.

    Detects dosage info masquerading as QL and normalizes to NA.
    """
    if _is_na(val):
        return "NA"

    lower = val.strip().lower()

    # Explicit
    if lower in ("no", "false", "no quantity limit", "no limits"):
        return "No"
    if lower in ("yes", "true"):
        return "Yes"

    # Dosage info, not QL
    dosage_signals = ["mg", "inject", "subcutaneous", "dose", "loading",
                      "maintenance", "every", "week"]
    ql_signals = ["quantity", "supply", "limit", "ql", "day supply",
                  "units per"]
    has_dosage = any(s in lower for s in dosage_signals)
    has_ql = any(s in lower for s in ql_signals)

    if has_dosage and not has_ql:
        return "NA"  # Dosage info, not quantity limits
    if has_ql:
        return "Yes"

    # Short affirmative
    if len(val.strip()) <= 5:
        return _normalize_yes_no(val)

    return "Yes" if len(val.strip()) > 20 else val.strip()


def normalize_extraction(params: dict) -> dict:
    """Normalize all extraction fields deterministically.

    Call this after every LLM extraction, before caching or scoring.
    Returns a new dict with normalized values.
    """
    result = dict(params)

    # Step therapy
    if "step_therapy_requirements" in result:
        result["step_therapy_requirements"] = _normalize_step_therapy(
            str(result["step_therapy_requirements"]))

    # Step counts
    for key in ("steps_through_brands", "steps_through_generic"):
        if key in result:
            result[key] = _normalize_step_count(str(result[key]))

    # Binary Yes/No fields
    for key in ("step_through_phototherapy", "tb_test_required",
                "reauth_required"):
        if key in result:
            result[key] = _normalize_yes_no(str(result[key]))

    # Quantity limits (special — detects dosage info)
    if "quantity_limits" in result:
        result["quantity_limits"] = _normalize_quantity_limits(
            str(result["quantity_limits"]))

    # Durations
    for key in ("initial_auth_duration", "reauth_duration"):
        if key in result:
            result[key] = _normalize_duration(str(result[key]))

    # Age
    if "age" in result:
        result["age"] = _normalize_age(str(result["age"]))

    # Specialist
    if "specialist_types" in result:
        result["specialist_types"] = _normalize_specialist(
            str(result["specialist_types"]))

    # Reauth requirements — free text, just clean up NA variants
    if "reauth_requirements" in result:
        val = str(result["reauth_requirements"]).strip()
        if _is_na(val):
            result["reauth_requirements"] = "NA"

    return result

"""
config.py — Constants, brand lists, column definitions, business rules.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from pipeline directory
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

# ── LLM Provider ───────────────────────────────────────────────────────
# "groq-8b-focused" (default): 8b-first extraction with 70b fallback
# "groq-70b-focused": per-parameter model routing (8b for simple, 70b for complex)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq-8b-focused")

# ── Groq (default) ────────────────────────────────────────────────────
# Accepts comma-separated keys for rotation.
_raw_groq = os.environ.get("GROQ_API_KEYS", "")
if not _raw_groq:
    _raw_groq = os.environ.get("GROQ_API_KEY", "")
GROQ_API_KEYS = [k.strip() for k in _raw_groq.split(",") if k.strip()]
GROQ_EXTRACTION_MODEL = os.environ.get("GROQ_EXTRACTION_MODEL", "llama-3.3-70b-versatile")
GROQ_VALIDATION_MODEL = os.environ.get("GROQ_VALIDATION_MODEL", "llama-3.1-8b-instant")

# ── Token budget constants ─────────────────────────────────────────
# Tokenization rate calibrated from actual Groq API responses on pharma
# policy text (drug names, medical terms, abbreviations tokenize poorly):
#   30,195 chars → 15,934 tokens (1.90 chars/token)
#   18,840 chars → 13,220 tokens (1.43 chars/token)
# Use 1.4 as conservative floor — overestimates tokens by 2-35%,
# which means we never hit 413s.
GROQ_CHARS_PER_TOKEN = float(os.environ.get("GROQ_CHARS_PER_TOKEN", "1.4"))
# Token budget per request (input + output combined).
# TPM is a rate limit (tokens/minute), not a per-request cap.
# A single request can use the full TPM; key rotation handles cooldown.
#   70b: 12,000 TPM   8b: 6,000 TPM
GROQ_TOKEN_BUDGET_70B = int(os.environ.get("GROQ_TOKEN_BUDGET_70B", "12000"))
GROQ_TOKEN_BUDGET_8B = int(os.environ.get("GROQ_TOKEN_BUDGET_8B", "6000"))
# Output token reservation.
GROQ_OUTPUT_TOKENS_8B = int(os.environ.get("GROQ_OUTPUT_TOKENS_8B", "500"))
GROQ_OUTPUT_TOKENS_70B = int(os.environ.get("GROQ_OUTPUT_TOKENS_70B", "2000"))

GROQ_MAX_CONTEXT_CHARS = int(os.environ.get("GROQ_MAX_CONTEXT_CHARS", "8000"))

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def compute_max_context_chars(prompt_chars: int, model: str = "70b",
                              natural_lang_overhead: int = 0) -> int:
    """Compute max context chars available for chunk text.

    Single-rate model: all content (prompts, UC, chunk text) tokenizes
    at GROQ_CHARS_PER_TOKEN (default 1.4 for pharma policy text).

    Args:
        prompt_chars: Fixed overhead (system prompt + template chrome) in chars.
        model: "8b" or "70b" selects the token budget tier.
        natural_lang_overhead: Additional fixed text sent with every request
            (e.g. universal criteria) — not part of the chunk.

    Returns at least 500 chars to avoid degenerate chunks.
    """
    r = GROQ_CHARS_PER_TOKEN
    is_8b = "8b" in model
    T = GROQ_TOKEN_BUDGET_8B if is_8b else GROQ_TOKEN_BUDGET_70B
    output_reserve = GROQ_OUTPUT_TOKENS_8B if is_8b else GROQ_OUTPUT_TOKENS_70B
    total_overhead_tokens = (prompt_chars + natural_lang_overhead) / r
    available_tokens = T - output_reserve - total_overhead_tokens
    # 5% safety margin for tokenization variance
    available_tokens *= 0.95
    available_chars = int(available_tokens * r)
    return max(available_chars, 500)
# Enable 8b model for simple parameters. Works on free tier with key rotation.
GROQ_USE_8B = os.environ.get("GROQ_USE_8B", "false").lower() in ("true", "1", "yes")

# ── Shared ─────────────────────────────────────────────────────────────
CALL_SPACING = float(os.environ.get("CALL_SPACING", "1"))  # Groq allows higher RPM

# ── Paths ──────────────────────────────────────────────────────────────
# Primary: relative to this file (works when running from the ZIP or repo).
# Fallback chain tries environment-specific locations.
_PIPELINE_ROOT = Path(__file__).resolve().parent

_PDF_DIR_CANDIDATES = [
    _PIPELINE_ROOT / "data" / "pdfs",
    Path("/workspaces/workspaces/ads_extracted/H1'26 ADS TT/data_extracted/Sample_PsO_ADS_Track"),
    Path.home() / "pipeline" / "data" / "pdfs",
]

_EXCEL_PATH_CANDIDATES = [
    _PIPELINE_ROOT / "data" / "PA_Business_Rules.xlsx",
    Path("/workspaces/workspaces/ads_extracted/H1'26 ADS TT/PA_Business_Rules.xlsx"),
    Path.home() / "pipeline" / "data" / "PA_Business_Rules.xlsx",
]

OUTPUT_DIR = _PIPELINE_ROOT / "output"
CACHE_DIR = str(_PIPELINE_ROOT / "cache")

# ── Context filter level ───────────────────────────────────────────────
# Controls how aggressively PDF text is filtered before sending to the LLM.
#   "page"      — original behaviour: include/exclude whole pages by relevance score
#   "paragraph" — split pages into paragraphs, keep only those matching PsO/brand/param keywords
#   "sentence"  — most aggressive: sentence-level extraction with 3-tier assembly
# Override via env var or CLI flag --filter-level.
FILTER_LEVEL = os.environ.get("FILTER_LEVEL", "sentence")

# Parameter-specific keyword patterns used by paragraph/sentence filters (Tier 3).
PARAMETER_PATTERNS: dict[str, list[str]] = {
    "age": ["years of age", "age of", ">=18", ">=6", ">=4", "pediatric",
            "adolescent", "adult patient", "years or older", "age restriction"],
    "step_therapy": ["step therapy", "tried and failed", "inadequate response",
                     "prior treatment", "failed to respond", "intolerance",
                     "contraindication", "conventional", "prerequisite",
                     "trial and failure", "trial of", "failed",
                     "must have tried", "documented failure", "treatment history",
                     "previous therapy", "prior use"],
    "branded_steps": ["preferred agent", "non-preferred", "biologic",
                      "biosimilar", "targeted immune", "targeted synthetic",
                      "preferred alternative", "formulary alternative"],
    "generic_steps": ["methotrexate", "cyclosporine", "acitretin", "topical",
                      "conventional synthetic", "non-biologic", "leflunomide",
                      "conventional therapy", "first-line"],
    "phototherapy": ["phototherapy", "puva", "uvb", "psoralen", "light therapy",
                     "light treatment", "narrowband"],
    "tb_test": ["tuberculosis", "tb test", "tb screen", "quantiferon",
                "latent tb", "tb testing", "ppd"],
    "quantity_limit": ["quantity limit", "quantity level limit",
                       "supply limit", "day supply"],
    "specialist": ["dermatologist", "specialist", "prescriber type",
                   "prescribed by", "in consultation with",
                   "board-certified", "treating physician"],
    "auth_duration": ["authorization duration", "approval period",
                      "authorized for", "approve for", "approved for",
                      "length of authorization", "initial authorization",
                      "valid for", "12 months", "6 months", "one year",
                      "365 days"],
    "reauth": ["reauthorization", "renewal", "continuation criteria",
               "continuation of therapy", "re-authorization",
               "continued approval", "renewal criteria",
               "subsequent authorization", "follow-up approval",
               "recertification"],
}

# Negative-filter keywords: paragraphs/sentences matching these AND NOT
# matching PsO signals are excluded (prevents PsA/UC/CD bleed).
NON_PSO_INDICATION_KEYWORDS = [
    "psoriatic arthritis", "ulcerative colitis", "crohn",
    "rheumatoid arthritis", "ankylosing spondylitis",
    "hidradenitis suppurativa",
]

# Substring-safe keywords for PsO detection.
# Ambiguous terms ("skin", "plaque", "bsa") replaced with specific phrases
# to avoid false positives (e.g. "skin test", "plaque buildup", "BSA" in
# non-dermatology contexts).
PSO_POSITIVE_KEYWORDS = [
    "plaque psoriasis", "psoriasis", "body surface area",
    "moderate-to-severe", "moderate to severe",
    "plaque pso",  # abbreviation variant
]

# Regex patterns for ambiguous PsO terms that need word-boundary matching.
# Used by has_pso_signal() below.
import re as _re
_PSO_REGEX_PATTERNS = [
    _re.compile(r'\bbsa\b'),       # "BSA" as standalone abbreviation
    _re.compile(r'\bplaque\b'),    # "plaque" not inside "plaque psoriasis" (already matched above)
]


def has_pso_signal(text_lower: str) -> bool:
    """Check if text contains a PsO-positive signal.

    Uses substring matching for unambiguous terms and regex word-boundary
    matching for ambiguous ones (bsa, plaque).
    """
    if any(kw in text_lower for kw in PSO_POSITIVE_KEYWORDS):
        return True
    return any(p.search(text_lower) for p in _PSO_REGEX_PATTERNS)


def get_pdf_dir() -> str:
    for candidate in _PDF_DIR_CANDIDATES:
        if candidate.is_dir():
            return str(candidate)
    # Return the primary path even if it doesn't exist yet (for error messages)
    return str(_PDF_DIR_CANDIDATES[0])


def get_excel_path() -> str:
    for candidate in _EXCEL_PATH_CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    return str(_EXCEL_PATH_CANDIDATES[0])


# Exact submission column names and order
SUBMISSION_COLUMNS = [
    "Filename",
    "Brand",
    "Age",
    "Step Therapy Requirements Documented in Policy",
    "Number of Steps through Brands",
    "Number of Steps through Generic",
    "Step through-Phototherapy",
    "TB Test required",
    "Quantity Limits",
    "Specialist Types",
    "Initial Authorization Duration(in-months)",
    "Reauthorization Duration(in-months)",
    "Reauthorization Required",
    "Reauthorization Requirements Documented in Policy",
    "Access Score",
]

# Biologic / branded drugs — count as "branded steps"
# Only originator biologics. Biosimilars are listed separately below.
BIOLOGIC_BRANDS = {
    "bimzelx", "cimzia", "cosentyx",
    "enbrel", "humira", "ilumya",
    "otezla", "remicade", "siliq", "skyrizi",
    "sotyktu", "stelara", "taltz", "tremfya", "simponi",
    "orencia", "actemra", "kevzara", "kineret", "entyvio", "rinvoq",
    "xeljanz", "olumiant", "tysabri",
}

# Biosimilars — interchangeable versions of reference biologics.
# Treated as generic steps, not branded steps, because policies that
# list "Avsola OR Inflectra OR Renflexis" are offering alternatives
# to the same reference product (infliximab), not separate step barriers.
BIOSIMILAR_BRANDS = {
    "amjevita", "avsola", "cyltezo", "hadlima", "hulio", "hyrimoz",
    "idacio", "imuldosa", "inflectra", "otulfi", "psychiva", "quallent",
    "renflexis", "selarsdi", "steqeyma", "wezlana", "yesintek",
    "yuflyma", "yusimry",
}

# Generic / non-biologic drugs — count as "generic steps"
GENERIC_DRUGS = {
    "methotrexate", "cyclosporine", "acitretin", "apremilast",
    "calcipotriene", "tazarotene", "anthralin", "coal tar",
    "salicylic acid", "sulfasalazine", "leflunomide",
    "hydroxychloroquine", "vtama", "zoryve",
}

# Brand name to generic name mapping
BRAND_TO_GENERIC = {
    "TREMFYA": "guselkumab",
    "STELARA": "ustekinumab",
    "ENBREL": "etanercept",
    "AMJEVITA": "adalimumab-atto",
    "COSENTYX": "secukinumab",
    "REMICADE": "infliximab",
    "SILIQ": "brodalumab",
    "CIMZIA": "certolizumab",
    "BIMZELX": "bimekizumab",
    "SKYRIZI": "risankizumab",
    "OTEZLA": "apremilast",
    "YESINTEK": "ustekinumab-kfce",
    "OTULFI": "ustekinumab-aauz",
    "ILUMYA": "tildrakizumab",
    "ACITRETIN": "acitretin",
}

# FDA labeled ages for psoriasis indication
FDA_LABELED_AGE = {
    "TREMFYA": ">=18",
    "STELARA": ">=6",
    "ENBREL": ">=4",
    "COSENTYX": ">=6",
    "AMJEVITA": ">=18",
    "REMICADE": ">=18",
    "SILIQ": ">=18",
    "CIMZIA": ">=18",
    "BIMZELX": ">=18",
    "SKYRIZI": ">=18",
    "OTEZLA": ">=18",
    "YESINTEK": ">=6",
    "OTULFI": ">=18",
    "ILUMYA": ">=18",
    "ACITRETIN": "No",
}

# Keywords for page-level relevance scoring
PRIMARY_KEYWORDS = [
    "psoriasis", "plaque", "pso",
]

BRAND_KEYWORDS = [
    "tremfya", "guselkumab", "stelara", "ustekinumab",
    "enbrel", "etanercept", "amjevita", "adalimumab",
    "cosentyx", "secukinumab", "remicade", "infliximab",
    "siliq", "brodalumab", "cimzia", "certolizumab",
    "bimzelx", "bimekizumab", "skyrizi", "risankizumab",
    "otezla", "apremilast", "yesintek", "otulfi",
    "ilumya", "tildrakizumab", "acitretin",
]

SECONDARY_KEYWORDS = [
    "step therapy", "prior treatment", "tried and failed",
    "inadequate response", "intolerance", "contraindication",
    "authorization duration", "approval period", "length of authorization",
    "reauthorization", "renewal", "continuation",
    "quantity limit", "quantity level limit",
    "specialist", "dermatologist", "prescriber",
    "tuberculosis", "tb test", "tb screen", "quantiferon", "latent tb",
    "phototherapy", "uvb", "puva", "psoralen",
]

UNIVERSAL_SECTION_KEYWORDS = [
    "general authorization", "general criteria", "all medications",
    "all indications", "non-preferred", "preferred agent",
    "applies to all", "for all requests", "for all diagnoses",
    "regardless of indication", "criteria for all",
]

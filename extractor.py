"""
extractor.py — Stage 3: LLM-based extraction.
Two-pass extraction (extract + validate) with caching, rate limiting,
and targeted JSON fix.

Two providers:
  groq-8b-focused: 8b-first extraction with 70b fallback (default)
  groq-70b-focused: per-parameter model routing (8b for simple, 70b for complex)

Groq uses two models:
  - Pass 1 (extraction): llama-3.3-70b-versatile
  - Pass 2 (validation): llama-3.1-8b-instant

Groq-Optimized routes per-parameter:
  - 7 simple params → llama-3.1-8b-instant (low token cost)
  - 5 complex params → llama-3.3-70b-versatile (higher accuracy)
  - 8b→70b escalation on validation failure
"""
import json
import os
import re
import time
import hashlib
from pathlib import Path

from normalizer import normalize_extraction
from regex_extractor import regex_extract, merge_regex_llm
from config import (
    CACHE_DIR, BRAND_TO_GENERIC, FDA_LABELED_AGE,
    LLM_PROVIDER, CALL_SPACING,
    GROQ_API_KEYS, GROQ_EXTRACTION_MODEL, GROQ_VALIDATION_MODEL, GROQ_BASE_URL,
    GROQ_TOKEN_BUDGET_8B, GROQ_OUTPUT_TOKENS_8B, GROQ_CHARS_PER_TOKEN,
    compute_max_context_chars,
    PARAMETER_PATTERNS,
)
from prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_COMPACT,
    SYSTEM_PROMPT_CHUNK,
    VALIDATION_SYSTEM_PROMPT,
    build_extraction_prompt,
    build_validation_prompt,
    select_few_shot_examples,
    format_few_shot_examples,
    MICRO_PROMPTS_8B,
    MICRO_SYSTEM_8B,
    MICRO_SYSTEM_70B,
    build_micro_prompt,
    build_batched_simple_prompt,
    build_complex_prompt,
    build_confirmation_prompt,
    CONFIRMATION_SYSTEM,
)
from step_analyzer import analyze_step_therapy
from pdf_extractor import ContextPackage


class RequestTooLarge(Exception):
    """Raised when a request exceeds the provider's token limit (413)."""
    pass


class AllKeysExhausted(Exception):
    """Raised when all API keys are unusable.

    Attributes:
        reason: "daily" if all keys hit daily limits (wait or add keys),
                "rate" if all keys hit per-minute limits after max retries,
                "dead" if all keys are invalid/restricted.
    """
    def __init__(self, reason: str, provider: str, message: str = ""):
        self.reason = reason
        self.provider = provider
        if not message:
            if reason == "daily":
                message = (
                    f"All {provider} API keys have hit their daily limit. "
                    f"Options:\n"
                    f"  1. Add more API keys in .env (GROQ_API_KEYS)\n"
                    f"  2. Wait until tomorrow for limits to reset\n"
                    f"  3. Upgrade to a paid plan for higher limits"
                )
            elif reason == "dead":
                message = (
                    f"All {provider} API keys are invalid or restricted. "
                    f"Check your API keys in .env and ensure they are active."
                )
            else:
                message = (
                    f"All {provider} API keys are rate-limited after multiple retry cycles. "
                    f"Try again in a few minutes, or add more API keys in .env."
                )
        super().__init__(message)


# Brand-specific defaults (Tier 3 fallback when all LLM calls fail)
BRAND_DEFAULTS = {
    "TREMFYA": {
        "age": "No",
        "step_therapy_requirements": "Unspecified",
        "steps_through_brands": "NA",
        "steps_through_generic": 1,
        "step_through_phototherapy": "No",
        "tb_test_required": "Y",
        "quantity_limits": "NA",
        "specialist_types": "NA",
        "initial_auth_duration": 12,
        "reauth_duration": "Unspecified",
        "reauth_required": "Yes",
        "reauth_requirements": "Unspecified",
    },
    "STELARA": {
        "age": ">=6",
        "step_therapy_requirements": "Unspecified",
        "steps_through_brands": "NA",
        "steps_through_generic": 1,
        "step_through_phototherapy": "No",
        "tb_test_required": "Y",
        "quantity_limits": "NA",
        "specialist_types": "NA",
        "initial_auth_duration": 12,
        "reauth_duration": "Unspecified",
        "reauth_required": "Yes",
        "reauth_requirements": "Unspecified",
    },
}

GENERIC_DEFAULTS = {
    "age": "No",
    "step_therapy_requirements": "Unspecified",
    "steps_through_brands": "NA",
    "steps_through_generic": 1,
    "step_through_phototherapy": "No",
    "tb_test_required": "Y",
    "quantity_limits": "NA",
    "specialist_types": "NA",
    "initial_auth_duration": 12,
    "reauth_duration": "Unspecified",
    "reauth_required": "Yes",
    "reauth_requirements": "Unspecified",
}


def _cache_key(filename: str, brand: str, pass_name: str,
               provider: str = "") -> str:
    """Generate a safe cache filename. Includes provider to avoid stale reads."""
    parts = f"{filename}_{brand}_{pass_name}"
    if provider:
        parts += f"_{provider}"
    safe_name = re.sub(r'[^\w\-.]', '_', parts)
    return safe_name + ".json"


# ── Shared merge logic ─────────────────────────────────────────────────

# Fields where "Yes" is more restrictive than "No"
_YES_NO_FIELDS = {
    "step_through_phototherapy", "tb_test_required",
    "quantity_limits", "reauth_required",
}

# Numeric fields where HIGHER = more restrictive (more steps = harder)
_HIGHER_RESTRICTIVE = {
    "steps_through_brands", "steps_through_generic",
}

# Numeric fields where LOWER = more restrictive (shorter auth = harder)
_LOWER_RESTRICTIVE = {
    "initial_auth_duration", "reauth_duration",
}

_NA_VALUES = {"na", "n/a", "not applicable", "not mentioned",
              "not specified", "not stated", "none", ""}


def _merge_results_field_aware(results: list[dict]) -> dict:
    """Merge multiple chunk extractions with field-aware conflict resolution.

    Rules:
    - Non-NA always wins over NA
    - Yes/No fields: "Yes" wins (more restrictive)
    - Step counts: higher value wins (more steps = harder access)
    - Durations: lower value wins (shorter auth = harder access)
    - Text fields: longest/most detailed wins
    - Reasoning: concatenated
    """
    if not results:
        return {}
    if len(results) == 1:
        return results[0]

    all_keys = set()
    for r in results:
        all_keys.update(r.keys())

    merged = {}

    for key in all_keys:
        values = [r.get(key, "NA") for r in results]
        substantive = [v for v in values
                       if str(v).strip().lower() not in _NA_VALUES]

        if not substantive:
            merged[key] = "NA"
            continue
        if len(substantive) == 1:
            merged[key] = substantive[0]
            continue

        # Multiple substantive values — field-aware resolution
        if key == "reasoning":
            merged[key] = " | ".join(str(v) for v in substantive)

        elif key in _YES_NO_FIELDS:
            # "Yes" is more restrictive
            if any(str(v).strip().lower() == "yes" for v in substantive):
                merged[key] = "Yes"
            else:
                merged[key] = substantive[0]

        elif key in _HIGHER_RESTRICTIVE:
            # Higher number = more restrictive
            nums = []
            for v in substantive:
                try:
                    nums.append((int(v), v))
                except (ValueError, TypeError):
                    pass
            if nums:
                merged[key] = str(max(nums, key=lambda x: x[0])[0])
            else:
                merged[key] = max(substantive, key=lambda v: len(str(v)))

        elif key in _LOWER_RESTRICTIVE:
            # Lower number = more restrictive (shorter auth)
            nums = []
            for v in substantive:
                try:
                    nums.append((int(v), v))
                except (ValueError, TypeError):
                    pass
            if nums:
                merged[key] = str(min(nums, key=lambda x: x[0])[0])
            else:
                merged[key] = max(substantive, key=lambda v: len(str(v)))

        elif key == "step_therapy_requirements":
            # Prefer "Yes" over "No", prefer long text over short
            has_yes = any(str(v).strip().lower() == "yes" for v in substantive)
            long_texts = [v for v in substantive if len(str(v)) > 20]
            if long_texts:
                merged[key] = max(long_texts, key=lambda v: len(str(v)))
            elif has_yes:
                merged[key] = "Yes"
            else:
                merged[key] = substantive[0]

        else:
            # Default: longest/most detailed value
            merged[key] = max(substantive, key=lambda v: len(str(v)))

    return merged


# ── Provider backends ──────────────────────────────────────────────────

class GroqRateLimitInfo:
    """Parsed rate limit info from Groq API error response.

    Groq headers don't distinguish per-minute vs per-day limits —
    `x-ratelimit-remaining-requests` is a single counter for the
    current window. We use `retry-after` duration to infer which
    limit was hit:
      - retry-after < 120s → per-minute (RPM/TPM), recoverable
      - retry-after >= 120s → likely per-day (RPD/TPD), kill the key
      - error message says "daily"/"per day" → definitely daily
    """
    # Threshold in seconds: retry-after above this = daily limit
    _DAILY_THRESHOLD = 120.0

    def __init__(self, error):
        self.retry_after = 0.0
        self.remaining_requests = None
        self.remaining_tokens = None
        self.is_daily_limit = False

        # Parse from error response headers
        try:
            headers = error.response.headers if hasattr(error, 'response') else {}
            self.retry_after = float(headers.get("retry-after", 0))
            self.remaining_requests = _safe_int(
                headers.get("x-ratelimit-remaining-requests"))
            self.remaining_tokens = _safe_int(
                headers.get("x-ratelimit-remaining-tokens"))

            # remaining_requests == 0 could be RPM or RPD.
            # Use retry-after to distinguish: short wait = per-minute,
            # long wait = per-day.
            if self.remaining_requests is not None and self.remaining_requests == 0:
                if self.retry_after >= self._DAILY_THRESHOLD:
                    self.is_daily_limit = True
                # else: per-minute exhaustion, will recover after cooldown
        except Exception:
            pass

        # Explicit daily limit signals in error message are definitive
        msg = str(error).lower()
        if "daily" in msg or "per day" in msg:
            self.is_daily_limit = True


def _safe_int(val):
    """Parse a header value to int, return None if not parseable."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _call_groq(prompt: str, system: str | None = None,
               model: str | None = None, api_key: str | None = None,
               call_spacing: float = 1.0,
               max_tokens: int | None = None) -> str:
    """Call Groq API using OpenAI-compatible SDK. Returns raw response text."""
    from openai import OpenAI

    key = api_key or (GROQ_API_KEYS[0] if GROQ_API_KEYS else "")
    if not key:
        raise RuntimeError("No GROQ_API_KEYS configured. Set it in .env")

    client = OpenAI(api_key=key, base_url=GROQ_BASE_URL)
    model_name = model or GROQ_EXTRACTION_MODEL

    # Default max_tokens based on model: 8b output is compact JSON (~300 tok),
    # 70b may include reasoning (~2000 tok).
    if max_tokens is None:
        max_tokens = 1024 if "8b" in (model_name or "") else 4096

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    time.sleep(call_spacing)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.1,
        top_p=0.95,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


class StandardExtractor:
    """8b-first extractor using Groq backend.

    Sends all 13 parameters in one prompt. Uses 8b for initial extraction
    attempt with 70b fallback, and 8b for validation.

    For per-parameter model routing, use OptimizedGroqExtractor instead.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 use_cache: bool = True, call_spacing: float | None = None,
                 provider: str | None = None, pdf_dir: str | None = None):
        self.provider = "groq"
        self.pdf_dir = pdf_dir
        self.use_cache = use_cache
        self.call_spacing = call_spacing if call_spacing is not None else CALL_SPACING
        self.api_calls_made = 0
        self.total_input_chars = 0
        self.total_output_chars = 0
        self.cache_hits = 0
        self.cache_dir = Path(CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manual_review_path = self.cache_dir.parent / "manual_review.txt"
        self._review_flags: list[str] = []  # accumulated per run, written at end

        if api_key:
            self._groq_keys = [k.strip() for k in api_key.split(",") if k.strip()]
        else:
            self._groq_keys = list(GROQ_API_KEYS)
        self._groq_key_index = 0
        self._extraction_model = GROQ_EXTRACTION_MODEL
        self._validation_model = GROQ_VALIDATION_MODEL
        if self._groq_keys:
            print(f"  Provider: Groq (8b-focused)")
            print(f"  Extraction model: {self._extraction_model}")
            print(f"  Validation model: {self._validation_model}")
        else:
            print("  WARNING: No GROQ_API_KEYS configured. Cache-only mode.")

        # Compact prompt with {brand} placeholder, formatted per-call
        self._is_compact = True
        self._system_prompt_template = SYSTEM_PROMPT_COMPACT
        self._system_prompt = None

    def _get_system_prompt(self, brand: str = "") -> str:
        """Get the system prompt, formatting brand placeholder if needed."""
        if self._system_prompt_template:
            return self._system_prompt_template.format(brand=brand)
        return self._system_prompt

    def _build_full_prompt(self, package, examples=None):
        """Build the complete user prompt (few-shot + extraction) for a package."""
        if examples is None:
            examples = select_few_shot_examples(package.brand,
                                                package.document_type)
        few_shot_text = format_few_shot_examples(examples,
                                                  compact=self._is_compact)
        extraction_prompt = build_extraction_prompt(package)
        return f"{few_shot_text}\n\n---\n\n{extraction_prompt}"

    @staticmethod
    def _parse_error_message(error: Exception) -> str:
        """Extract a clean one-line message from a provider error."""
        s = str(error)
        # Groq/OpenAI errors contain a JSON 'message' field — extract it
        import re as _re
        m = _re.search(r"'message':\s*'([^']{10,}?)'", s)
        if m:
            msg = m.group(1)
            # Trim the "Need more tokens? Upgrade..." marketing suffix
            cut = msg.find("Need more tokens?")
            if cut > 0:
                msg = msg[:cut].rstrip(". ")
            return msg
        # Fallback: first 150 chars
        return s[:150]

    def _call_llm(self, prompt: str, system: str | None = None,
                  pass_type: str = "extract") -> str:
        """Groq LLM call with retry logic.

        Args:
            pass_type: "extract" or "validate" — controls model selection.

        Key rotation strategy (matches OptimizedGroqExtractor):
        - Round-robin before each call
        - Daily limit: remove key permanently for this run
        - Per-minute limit: rotate to next key, short wait
        - Full cycle (all keys rate-limited): wait retry-after (capped 10s),
          then try another cycle. Max 2 full cycles before giving up.
        """
        max_cycles = 2
        max_attempts = max(max_cycles * len(self._groq_keys), 3)
        failures_this_cycle = 0
        server_errors = 0

        for attempt in range(max_attempts):
            try:
                if len(self._groq_keys) > 1:
                    self._groq_key_index = (self._groq_key_index + 1) % len(self._groq_keys)
                model = (self._extraction_model if pass_type == "extract"
                         else self._validation_model)
                key = self._groq_keys[self._groq_key_index]
                text = _call_groq(
                    prompt, system=system, model=model,
                    api_key=key,
                    call_spacing=self.call_spacing,
                )

                self.api_calls_made += 1
                input_chars = len(prompt) + (len(system) if system else 0)
                self.total_input_chars += input_chars
                self.total_output_chars += len(text)
                return text

            except Exception as e:
                error_str = str(e).lower()
                clean_msg = self._parse_error_message(e)

                # 413 = request too large — retrying won't help
                if "413" in error_str or "request too large" in error_str:
                    print(f"    [413] Request too large: {clean_msg}")
                    raise RequestTooLarge(clean_msg)

                if "429" in error_str or "rate" in error_str or "quota" in error_str:
                    rate_info = GroqRateLimitInfo(e)

                    # Daily limit — remove key for this run
                    if rate_info.is_daily_limit:
                        old = self._groq_key_index
                        self._groq_keys.pop(old)
                        if self._groq_keys:
                            self._groq_key_index = old % len(self._groq_keys)
                            print(f"    [DAILY] key #{old+1} hit daily limit, "
                                  f"removed ({len(self._groq_keys)} remaining)")
                        else:
                            raise AllKeysExhausted("daily", "groq")
                        failures_this_cycle = 0
                        continue

                    # Per-minute limit — rotate and track cycle
                    failures_this_cycle += 1

                    if failures_this_cycle >= len(self._groq_keys):
                        # Full cycle: all live keys hit rate limit — cooldown
                        failures_this_cycle = 0
                        wait = rate_info.retry_after if rate_info.retry_after > 0 else 3.0
                        wait = min(wait, 10.0)
                        print(f"    [COOLDOWN] all {len(self._groq_keys)} keys "
                              f"rate-limited, waiting {wait:.0f}s")
                        time.sleep(wait)
                    else:
                        # Rotate to next key
                        if len(self._groq_keys) > 1:
                            old = self._groq_key_index
                            self._groq_key_index = (old + 1) % len(self._groq_keys)
                            print(f"    [429] key #{old+1} -> "
                                  f"#{self._groq_key_index+1}")
                        else:
                            # Single key — wait before retry
                            wait = rate_info.retry_after if rate_info.retry_after > 0 else 3.0
                            wait = min(wait, 10.0)
                            print(f"    [429] single key, waiting {wait:.0f}s")
                            time.sleep(wait)

                elif "500" in error_str or "internal" in error_str:
                    server_errors += 1
                    if server_errors >= 3:
                        print(f"    [500] 3 server errors, giving up")
                        return ""
                    wait = self.call_spacing * server_errors
                    print(f"    [500] Server error ({server_errors}/3), "
                          f"retrying in {wait:.0f}s")
                    time.sleep(wait)
                else:
                    print(f"    [ERR] {clean_msg}")
                    return ""

        raise AllKeysExhausted("rate", "groq")

    # ── Caching ────────────────────────────────────────────────────────

    def _load_cache(self, key: str) -> dict | None:
        if not self.use_cache:
            return None
        cache_path = self.cache_dir / key
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                self.cache_hits += 1
                return data
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def _save_cache(self, key: str, data: dict):
        # Normalize extraction output before persisting
        data = normalize_extraction(data)
        cache_path = self.cache_dir / key
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)

    # ── JSON parsing ──────────────────────────────────────────────────

    def _parse_json_response(self, response_text: str) -> dict | None:
        if not response_text:
            return None

        # Direct parse
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Markdown code block
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # First { ... } block
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _fix_json(self, broken_text: str) -> dict | None:
        """Send a targeted fix prompt for malformed JSON. Costs 1 API call."""
        fix_prompt = (
            "The following text was supposed to be valid JSON but has syntax "
            "errors.\nFix the JSON syntax and return ONLY the corrected JSON "
            "object. Do not change any values, only fix syntax.\n\n"
            f"Broken text:\n{broken_text[:4000]}"
        )
        try:
            response = self._call_llm(
                fix_prompt,
                system="You fix JSON syntax errors. Return only valid JSON.",
                pass_type="validate",
            )
            return self._parse_json_response(response)
        except RequestTooLarge:
            # JSON fix is a nice-to-have — don't let it trigger chunk splitting
            return None

    def _log_manual_review(self, filename: str, brand: str, reason: str):
        self._review_flags.append(f"{filename} | {brand} | {reason}")

    def _flush_manual_review(self):
        """Write accumulated review flags to file (replaces previous content).

        Called once at the end of a batch run. If no flags, deletes the file.
        """
        if self._review_flags:
            with open(self.manual_review_path, "w") as f:
                f.write("\n".join(self._review_flags) + "\n")
        elif self.manual_review_path.exists():
            self.manual_review_path.unlink()

    def _clear_manual_review(self):
        """Reset review flags for a new run."""
        self._review_flags = []

    # ── Chunked extraction for large contexts ──────────────────────────

    def _compute_chunk_budgets(self, package: ContextPackage) -> tuple[int, int]:
        """Compute chunk budgets for 70b: (budget_with_uc, budget_without_uc).

        UC text is only sent with the first chunk. Subsequent chunks get
        a larger budget since they don't carry UC overhead.
        """
        dummy_pkg = ContextPackage(
            filename=package.filename, brand=package.brand,
            preferred_status=package.preferred_status,
            document_type=package.document_type,
            total_pages=package.total_pages,
            relevant_pages_used=package.relevant_pages_used,
            full_relevant_text="",
            universal_criteria_text="",
        )
        extraction_prompt = build_extraction_prompt(dummy_pkg)
        chunk_system = SYSTEM_PROMPT_CHUNK.format(brand=package.brand)
        base_overhead = len(extraction_prompt) + len(chunk_system)
        uc_chars = len(package.universal_criteria_text or "")
        model = GROQ_EXTRACTION_MODEL
        budget_with_uc = compute_max_context_chars(
            base_overhead, model, natural_lang_overhead=uc_chars)
        budget_no_uc = compute_max_context_chars(
            base_overhead, model, natural_lang_overhead=0)
        return budget_with_uc, budget_no_uc

    def _needs_chunking(self, package: ContextPackage) -> bool:
        """Check if context exceeds what the full prompt can handle.

        Tests against the FULL prompt (with few-shot + full system prompt),
        not the compact chunk prompt. If the full prompt doesn't fit,
        we switch to chunked extraction with compact prompts.
        """
        # Measure structured overhead with empty UC and empty context
        dummy_pkg = ContextPackage(
            filename=package.filename, brand=package.brand,
            preferred_status=package.preferred_status,
            document_type=package.document_type,
            total_pages=package.total_pages,
            relevant_pages_used=package.relevant_pages_used,
            full_relevant_text="",
            universal_criteria_text="",
        )
        system = self._get_system_prompt(package.brand)
        structured_overhead = len(self._build_full_prompt(dummy_pkg))
        structured_overhead += len(system)
        uc_chars = len(package.universal_criteria_text or "")
        full_budget = compute_max_context_chars(
            structured_overhead, GROQ_EXTRACTION_MODEL,
            natural_lang_overhead=uc_chars)
        return len(package.full_relevant_text) > full_budget

    @staticmethod
    def _split_into_chunks(text: str, max_chars: int) -> list[str]:
        """Split context text on section markers, keeping chunks under max_chars.

        Splits on [Universal | Page N], [PsO Section | Page N], etc.
        Falls back to paragraph boundaries if no markers found.
        """
        # Split on section markers inserted by pdf_extractor
        parts = re.split(r'(?=\[(?:Universal|PsO Section|Tier \d|Page)[^\]]*\])',
                         text)
        parts = [p for p in parts if p.strip()]

        if len(parts) <= 1:
            # No markers — split on double-newlines (paragraph boundaries)
            parts = re.split(r'\n{2,}', text)
            parts = [p for p in parts if p.strip()]

        # Sub-split any individual part that exceeds max_chars
        final_parts = []
        for part in parts:
            if len(part) <= max_chars:
                final_parts.append(part)
            else:
                # Split oversized part on paragraph boundaries
                sub = re.split(r'\n{2,}', part)
                sub = [s for s in sub if s.strip()]
                if not sub:
                    # Hard split as last resort
                    for i in range(0, len(part), max_chars):
                        final_parts.append(part[i:i + max_chars])
                else:
                    final_parts.extend(sub)

        chunks = []
        current = ""
        for part in final_parts:
            if current and len(current) + len(part) > max_chars:
                chunks.append(current)
                current = part
            else:
                current = current + part if current else part
        if current:
            chunks.append(current)

        return chunks if chunks else [text[:max_chars]]

    def _merge_extractions(self, results: list[dict], brand: str) -> dict:
        """Merge multiple chunk extractions with field-aware conflict resolution.

        Non-NA always wins over NA. For conflicts between substantive values:
        - Yes/No fields: "Yes" wins (more restrictive)
        - Numeric fields (steps, durations): most restrictive wins
        - Text fields: longest/most detailed wins
        - Reasoning: concatenated
        """
        return _merge_results_field_aware(results)

    @staticmethod
    def _merge_extractions_static(results: list[dict]) -> dict:
        return _merge_results_field_aware(results)

    def _extract_one_chunk(self, chunk_text: str, package: ContextPackage,
                           chunk_label: str, is_first: bool,
                           _depth: int = 0) -> list[dict]:
        """Extract from a single chunk. On RequestTooLarge, split in half and retry.

        Uses compact system prompt and no few-shot examples to minimize
        overhead and maximize context budget per chunk.

        Returns a list of result dicts (may be >1 if the chunk was re-split).
        """
        chunk_pkg = ContextPackage(
            filename=package.filename,
            brand=package.brand,
            preferred_status=package.preferred_status,
            document_type=package.document_type,
            total_pages=package.total_pages,
            relevant_pages_used=package.relevant_pages_used,
            full_relevant_text=chunk_text,
            universal_criteria_text=(
                package.universal_criteria_text if is_first else ""
            ),
        )

        # Use extraction template only (no few-shot) to keep overhead low
        chunk_prompt = build_extraction_prompt(chunk_pkg)
        chunk_system = SYSTEM_PROMPT_CHUNK.format(brand=package.brand)

        # 70b extraction (8b is handled at the _extract_chunked level)
        try:
            for attempt in range(2):
                response = self._call_llm(chunk_prompt,
                                          system=chunk_system,
                                          pass_type="extract")
                result = self._parse_json_response(response)
                if result is not None:
                    print(f"    {chunk_label}: OK (70b)")
                    return [result]
                if response:
                    result = self._fix_json(response)
                    if result is not None:
                        print(f"    {chunk_label}: OK (70b, JSON fix)")
                        return [result]
        except RequestTooLarge:
            if len(chunk_text) < 500 or _depth >= 3:
                print(f"    {chunk_label}: SKIP (too large even at "
                      f"{len(chunk_text):,} chars, depth {_depth})")
                return []
            # Split in half and retry each sub-chunk
            mid = len(chunk_text) // 2
            nl = chunk_text.rfind('\n', mid - 200, mid + 200)
            if nl > 0:
                mid = nl
            print(f"    {chunk_label}: too large, splitting "
                  f"({len(chunk_text):,} -> {mid:,} + {len(chunk_text)-mid:,})")
            left = self._extract_one_chunk(
                chunk_text[:mid], package,
                f"{chunk_label}a", is_first, _depth + 1)
            right = self._extract_one_chunk(
                chunk_text[mid:], package,
                f"{chunk_label}b", False, _depth + 1)
            return left + right

        print(f"    {chunk_label}: FAIL (bad response)")
        return []

    def _compute_8b_chunk_budgets(self, package: ContextPackage) -> tuple[int, int]:
        """Compute chunk budgets for 8b: (budget_with_uc, budget_without_uc)."""
        dummy_pkg = ContextPackage(
            filename=package.filename, brand=package.brand,
            preferred_status=package.preferred_status,
            document_type=package.document_type,
            total_pages=package.total_pages,
            relevant_pages_used=package.relevant_pages_used,
            full_relevant_text="",
            universal_criteria_text="",
        )
        extraction_prompt = build_extraction_prompt(dummy_pkg)
        compact_system = SYSTEM_PROMPT_COMPACT.format(brand=package.brand)
        base_overhead = len(extraction_prompt) + len(compact_system)
        uc_chars = len(package.universal_criteria_text or "")
        budget_with_uc = compute_max_context_chars(
            base_overhead, GROQ_VALIDATION_MODEL, natural_lang_overhead=uc_chars)
        budget_no_uc = compute_max_context_chars(
            base_overhead, GROQ_VALIDATION_MODEL, natural_lang_overhead=0)
        return budget_with_uc, budget_no_uc

    def _extract_chunked(self, package: ContextPackage) -> dict | None:
        """Chunked extraction: 8b for most chunks, 70b only when needed.

        Strategy:
        - Chunk 0 (carries UC): send to 70b directly (UC makes it too large
          for 8b in most cases, and a failed 8b attempt wastes rate limit)
        - Chunks 1+ (no UC): try 8b first. On 413 (too large), fall back
          to 70b for that chunk. On rate limit, retry with key rotation.
        """
        _, b8_no = self._compute_8b_chunk_budgets(package)
        b70_uc, b70_no = self._compute_chunk_budgets(package)
        uc_len = len(package.universal_criteria_text or "")
        text = package.full_relevant_text
        text_len = len(text)

        # Split chunk 0 at 70b UC budget, rest at 8b no-UC budget
        chunk0_text = text[:b70_uc]
        rest_text = text[b70_uc:]

        # Chunk 0 may not align to a clean boundary — find nearest newline
        if rest_text:
            nl = text.rfind('\n', max(0, b70_uc - 200), b70_uc + 200)
            if nl > 0:
                chunk0_text = text[:nl]
                rest_text = text[nl:]

        rest_chunks = self._split_into_chunks(rest_text, b8_no) if rest_text else []
        total_chunks = 1 + len(rest_chunks)

        print(f"    Chunked extraction: {text_len:,} chars, UC={uc_len:,}")
        print(f"      chunk0: {len(chunk0_text):,} chars -> 70b (carries UC)")
        print(f"      rest: {len(rest_chunks)} chunks @ {b8_no:,} -> 8b first")

        chunk_results = []

        # ── Chunk 0: always 70b (carries UC) ──
        results = self._extract_one_chunk(
            chunk0_text, package, f"Chunk 1/{total_chunks}",
            is_first=True)
        chunk_results.extend(results)

        # ── Chunks 1+: try 8b, per-chunk 70b fallback ──
        for i, chunk_text in enumerate(rest_chunks):
            label = f"Chunk {i+2}/{total_chunks}"

            chunk_pkg = ContextPackage(
                filename=package.filename, brand=package.brand,
                preferred_status=package.preferred_status,
                document_type=package.document_type,
                total_pages=package.total_pages,
                relevant_pages_used=package.relevant_pages_used,
                full_relevant_text=chunk_text,
                universal_criteria_text="",  # no UC for chunks 1+
            )
            result_8b = self._try_8b_extract(chunk_pkg)

            if isinstance(result_8b, dict):
                print(f"    {label}: OK (8b)")
                chunk_results.append(result_8b)
                continue

            if result_8b == "too_large":
                # 8b can't handle this chunk size — use 70b
                print(f"    {label}: 8b too large, using 70b")
            else:
                # Rate limit or other transient error — still try 70b
                # (8b keys may be exhausted but 70b keys are separate)
                print(f"    {label}: 8b unavailable, using 70b")

            results = self._extract_one_chunk(
                chunk_text, package, label, is_first=False)
            chunk_results.extend(results)

        if not chunk_results:
            print(f"    Result: ALL CHUNKS FAILED")
            return None

        merged = self._merge_extractions(chunk_results, package.brand)
        print(f"    Result: merged {len(chunk_results)} chunk(s) "
              f"-> {sum(1 for v in merged.values() if str(v).strip().upper() not in ('NA','N/A',''))} "
              f"non-NA fields")
        return merged

    # ── 8b keyword-scored extraction ────────────────────────────────

    _8B_KEYWORDS = [
        # Indication
        "plaque psoriasis", "psoriasis", "moderate to severe",
        # Step therapy
        "step therapy", "trial and failure", "tried and failed",
        "must have tried", "conventional therapy", "systemic therapy",
        "biologic", "non-biologic",
        # Specific treatments in step therapy
        "methotrexate", "cyclosporine", "acitretin", "phototherapy",
        "puva", "uvb", "topical",
        # TB
        "tb test", "tuberculosis", "latent tb",
        # Authorization
        "initial authorization", "authorization duration", "coverage duration",
        "reauthorization", "renewal criteria", "continuation criteria",
        # Other params
        "quantity limit", "quantity limits",
        "prior authorization", "coverage criteria", "approval criteria",
        "age restriction", "prescriber restriction",
        "specialist", "dermatologist",
    ]

    def _select_relevant_chunks(self, text: str, brand: str,
                                max_chars: int = 10000) -> str:
        """Select most relevant chunks using keyword scoring.

        For small texts (< max_chars), returns the full text directly.
        For large texts, scores 3K chunks by keyword hits and selects
        the top-scoring chunks plus neighbors for context continuity.
        """
        # Small text — no need to chunk, send it all
        if len(text) <= max_chars:
            return text

        brand_lower = brand.lower()
        keywords = self._8B_KEYWORDS + [brand_lower]
        from config import BRAND_TO_GENERIC
        generic = BRAND_TO_GENERIC.get(brand.upper(), "")
        if generic:
            keywords.append(generic.lower())

        chunk_size, overlap = 3000, 250
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = end - overlap

        # Score each chunk by keyword hits
        scored = []
        for i, chunk in enumerate(chunks):
            low = chunk.lower()
            score = sum(1 for kw in keywords if kw in low)
            if score > 0:
                scored.append((score, i))

        if not scored:
            return "\n\n".join(chunks[:3])[:max_chars]

        scored.sort(key=lambda x: (-x[0], x[1]))

        # Select top chunks + neighbors for context continuity
        selected = set()
        for _, idx in scored[:8]:
            for n in range(max(0, idx - 1), min(len(chunks), idx + 2)):
                selected.add(n)

        context = "\n\n".join(chunks[i] for i in sorted(selected))
        return context[:max_chars]

    def _build_8b_field_hints(self, brand: str) -> str:
        """Build field-specific extraction hints for 8b.

        Format-agnostic: works across structured PA criteria, clinical policy
        bulletins, drug policies, and narrative coverage documents.
        """
        return f"""Extract prior authorization policy parameters for {brand} for Plaque Psoriasis (PsO).
Return valid JSON only. Use "NA" if the parameter is not mentioned anywhere in the text.

Fields:
- age: patient age requirement. Use exact text (e.g. ">=18", ">=6", "adults"). "NA" if none stated
- step_therapy_requirements: what therapies must be tried/failed before {brand} is approved. Quote policy language. "NA" if none
- steps_through_brands: number of biologic/branded drugs required before {brand}. Integer or "NA"
- steps_through_generic: number of non-biologic therapies (methotrexate, cyclosporine, topicals, etc.) required. Integer or "NA"
- step_through_phototherapy: "Yes" if phototherapy is a mandatory prerequisite (not just one option among alternatives). "No" if optional or in an OR list. "N/A" if not mentioned
- tb_test_required: "Y" if TB/tuberculosis testing is required. "NA" if not mentioned
- quantity_limits: explicit quantity limits only (not dosing schedules). "NA" if none
- specialist_types: which specialists can prescribe (e.g. "Dermatologist"). "NA" if not restricted
- initial_auth_duration: initial approval period in months. Integer or "NA"
- reauth_duration: renewal/reauthorization period in months. Integer or "NA"
- reauth_required: "Yes" if renewal criteria or reauth duration exist. "No" otherwise
- reauth_requirements: what is needed for renewal (e.g. clinical response documentation). "NA" if none
- reasoning: 1 sentence summary of the policy's access restrictions"""

    def _try_8b_extract(self, package: ContextPackage) -> dict | str | None:
        """Try extraction with 8b using keyword-scored context selection.

        Selects the most relevant chunks from the document using keyword
        scoring, then sends a focused prompt with field-specific hints.
        Returns dict on success, "too_large" on 413, None on transient error.
        """
        # Select relevant context via keyword scoring
        full_text = package.full_relevant_text
        uc_text = package.universal_criteria_text or ""
        combined = (uc_text + "\n\n" + full_text) if uc_text else full_text

        # Budget: 8b has 6K TPM, max_tokens=1200 → 4800 input tokens.
        # At ~2.5 ch/tok (typical pharma text), that's ~12K chars total.
        # Hints+system ≈ 1200 chars, leaving ~10K for context.
        context = self._select_relevant_chunks(combined, package.brand,
                                               max_chars=10000)

        field_hints = self._build_8b_field_hints(package.brand)
        user_prompt = (f"{field_hints}\n\n"
                       f"Policy text:\n{context}")
        system = "You extract structured policy data. Return JSON only."
        total_chars = len(user_prompt) + len(system)

        # Sanity check: skip if even optimistic estimate won't fit
        if total_chars / 3.5 > (GROQ_TOKEN_BUDGET_8B - GROQ_OUTPUT_TOKENS_8B):
            print(f"    [8b skip] {total_chars:,} chars too large")
            return None

        try:
            if len(self._groq_keys) > 1:
                self._groq_key_index = (self._groq_key_index + 1) % len(self._groq_keys)
            key = self._groq_keys[self._groq_key_index]

            print(f"    [8b] keyword-scored {total_chars:,} chars "
                  f"(context={len(context):,} from {len(combined):,})")

            response = _call_groq(
                user_prompt, system=system,
                model=self._validation_model,  # 8b
                api_key=key,
                call_spacing=self.call_spacing,
                max_tokens=1200,
            )
            self.api_calls_made += 1
            self.total_input_chars += total_chars
            self.total_output_chars += len(response)

            result = self._parse_json_response(response)
            if result is None and response:
                result = self._fix_json(response)

            if result is not None:
                print(f"    [{package.brand}] OK (8b)")
            return result

        except Exception as e:
            error_str = str(e).lower()
            if "413" in error_str or "too large" in error_str:
                print(f"    [8b too large] falling back to 70b")
                return "too_large"
            elif "429" in error_str or "rate" in error_str:
                print(f"    [8b rate-limited] rotating key")
                return None
            else:
                print(f"    [8b failed] {str(e)[:80]}")
                return None

    # ── Extraction ────────────────────────────────────────────────────

    def extract_single(self, package: ContextPackage) -> dict:
        """
        Extract parameters for a single (PDF, Brand) pair.
        Uses chunked extraction for oversized contexts on Groq.
        Three-tier fallback: 8b -> 70b -> simplified -> defaults.
        """
        cache_key = _cache_key(package.filename, package.brand, "extract", self.provider)
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        # --- Regex first pass (no LLM, instant) ---
        regex_result = None
        if self.pdf_dir:
            pdf_path = os.path.join(self.pdf_dir, package.filename)
            if os.path.exists(pdf_path):
                try:
                    regex_result = regex_extract(pdf_path, package.brand)
                except Exception as e:
                    print(f"    [regex] failed: {str(e)[:80]}")

        # Check if regex got everything (all 12 params non-None)
        if regex_result:
            filled = sum(1 for v in regex_result.values() if v is not None)
            if filled == 12:
                print(f"  [{package.brand}] regex extracted all 12 params, skipping LLM")
                self._save_cache(cache_key, regex_result)
                return regex_result

        ctx_chars = len(package.full_relevant_text)
        needs_chunking = self._needs_chunking(package)

        # --- Chunked extraction for oversized Groq contexts ---
        if needs_chunking:
            print(f"  [{package.brand}] {package.filename} "
                  f"({ctx_chars:,} chars) -> chunked LLM analysis")
            result = self._extract_chunked(package)
            if result is not None:
                if regex_result:
                    result = merge_regex_llm(regex_result, result)
                self._save_cache(cache_key, result)
                return result
            print(f"  [{package.brand}] Chunked failed, trying Tier 2")

        system_prompt = self._get_system_prompt(package.brand)
        full_prompt = self._build_full_prompt(package)
        total_chars = len(full_prompt) + len(system_prompt)

        # --- Tier 0: Try 8b first ---
        print(f"  [{package.brand}] {package.filename} "
              f"({ctx_chars:,} chars) -> trying 8b first")
        result = self._try_8b_extract(package)
        if isinstance(result, dict):
            if regex_result:
                result = merge_regex_llm(regex_result, result)
            self._save_cache(cache_key, result)
            return result

        # Helper: merge regex results into LLM result before saving
        def _save(result):
            if regex_result:
                result = merge_regex_llm(regex_result, result)
            self._save_cache(cache_key, result)
            return result

        # --- Tier 1: Standard extraction (70b) ---
        if not needs_chunking:
            try:
                for attempt in range(3):
                    response = self._call_llm(full_prompt,
                                              system=system_prompt,
                                              pass_type="extract")
                    result = self._parse_json_response(response)
                    if result is not None:
                        print(f"  [{package.brand}] OK (70b)")
                        return _save(result)

                    if response:
                        result = self._fix_json(response)
                        if result is not None:
                            print(f"  [{package.brand}] OK (70b, JSON fix)")
                            return _save(result)

                    print(f"    Tier 1 attempt {attempt + 1}/3 failed")
            except RequestTooLarge:
                print(f"  [{package.brand}] Too large for direct, "
                      f"falling back to chunked")
                result = self._extract_chunked(package)
                if result is not None:
                    return _save(result)

        # --- Tier 2: Simplified prompt (truncated to fit) ---
        print(f"  [{package.brand}] Trying Tier 2 (simplified prompt)")
        _tier2_overhead = 400 + len(system_prompt)
        max_text = compute_max_context_chars(_tier2_overhead, GROQ_EXTRACTION_MODEL)
        simple_prompt = (
            f"Extract these 13 PA policy parameters for {package.brand} "
            f"for Plaque Psoriasis from the text below. "
            f"Return valid JSON only.\n\n"
            f"Parameters: age, step_therapy_requirements, "
            f"steps_through_brands, steps_through_generic, "
            f"step_through_phototherapy, tb_test_required, "
            f"quantity_limits, specialist_types, initial_auth_duration, "
            f"reauth_duration, reauth_required, reauth_requirements, "
            f"reasoning\n\n"
            f"Policy text:\n{package.full_relevant_text[:max_text]}"
        )
        try:
            for attempt in range(2):
                response = self._call_llm(simple_prompt,
                                          system=system_prompt,
                                          pass_type="extract")
                result = self._parse_json_response(response)
                if result is not None:
                    print(f"  [{package.brand}] OK (Tier 2)")
                    return _save(result)
        except RequestTooLarge:
            try:
                simple_prompt = simple_prompt.replace(
                    package.full_relevant_text[:max_text],
                    package.full_relevant_text[:max_text // 2]
                )
                response = self._call_llm(simple_prompt,
                                          system=system_prompt,
                                          pass_type="extract")
                result = self._parse_json_response(response)
                if result is not None:
                    print(f"  [{package.brand}] OK (Tier 2, truncated)")
                    return _save(result)
            except RequestTooLarge:
                pass

        # --- Tier 3: Brand-specific defaults (merge regex if available) ---
        print(f"  [{package.brand}] FALLBACK to brand defaults")
        self._log_manual_review(
            package.filename, package.brand,
            "All extraction tiers failed — using brand defaults"
        )
        defaults = dict(BRAND_DEFAULTS.get(package.brand, GENERIC_DEFAULTS))
        return _save(defaults)

    def validate_single(self, extraction: dict,
                        package: ContextPackage) -> dict:
        """
        Validate and correct a single extraction against the source text.
        Uses the lighter validation model on Groq.
        """
        cache_key = _cache_key(package.filename, package.brand, "validate", self.provider)
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        # Compute actual overhead from template + extraction JSON (no source text)
        extraction_json = json.dumps(extraction, indent=2)
        empty_prompt = build_validation_prompt(
            brand=package.brand, filename=package.filename,
            extracted_json=extraction_json, source_text="",
        )
        val_overhead = len(empty_prompt) + len(VALIDATION_SYSTEM_PROMPT)

        # Try 8b first; if source text doesn't fit, fall back to 70b
        max_source_8b = compute_max_context_chars(val_overhead, GROQ_VALIDATION_MODEL)
        if len(package.full_relevant_text) <= max_source_8b:
            val_pass_type = "validate"  # 8b
            max_source = max_source_8b
        else:
            val_pass_type = "extract"   # 70b
            max_source = compute_max_context_chars(val_overhead, GROQ_EXTRACTION_MODEL)
            print(f"  [{package.brand}] Validation: source too large for 8b "
                  f"({len(package.full_relevant_text):,} > {max_source_8b:,}), "
                  f"using 70b (budget {max_source:,})")

        source_text = package.full_relevant_text[:max_source]

        validation_prompt = build_validation_prompt(
            brand=package.brand,
            filename=package.filename,
            extracted_json=extraction_json,
            source_text=source_text,
        )

        val_model = (self._validation_model if val_pass_type == "validate"
                     else self._extraction_model)
        val_total = len(validation_prompt) + len(VALIDATION_SYSTEM_PROMPT)
        print(f"  [{package.brand}] Validating: source={len(source_text):,}/{len(package.full_relevant_text):,} chars, "
              f"prompt={val_total:,} chars, model={val_model}")

        try:
            for attempt in range(2):
                response = self._call_llm(
                    validation_prompt,
                    system=VALIDATION_SYSTEM_PROMPT,
                    pass_type=val_pass_type,
                )
                result = self._parse_json_response(response)
                if result is not None:
                    # Count corrections
                    corrections = sum(
                        1 for k in extraction
                        if k in result and str(extraction[k]) != str(result[k])
                    )
                    if corrections:
                        print(f"  [{package.brand}] Validated ({corrections} correction(s))")
                    else:
                        print(f"  [{package.brand}] Validated (no changes)")
                    self._save_cache(cache_key, result)
                    return result

                if response:
                    result = self._fix_json(response)
                    if result is not None:
                        print(f"  [{package.brand}] Validated (after JSON fix)")
                        self._save_cache(cache_key, result)
                        return result
        except RequestTooLarge:
            # Retry with further truncated text
            try:
                validation_prompt = build_validation_prompt(
                    brand=package.brand,
                    filename=package.filename,
                    extracted_json=json.dumps(extraction, indent=2),
                    source_text=source_text[:len(source_text) // 2],
                )
                response = self._call_llm(
                    validation_prompt,
                    system=VALIDATION_SYSTEM_PROMPT,
                    pass_type=val_pass_type,
                )
                result = self._parse_json_response(response)
                if result is not None:
                    print(f"  [{package.brand}] Validated (truncated source)")
                    self._save_cache(cache_key, result)
                    return result
            except RequestTooLarge:
                pass

        print(f"  [{package.brand}] Validation skipped (keeping extraction)")
        return extraction

    # ── Batch operations ──────────────────────────────────────────────

    def extract_all(self, packages: list[ContextPackage]) -> list[dict]:
        """Run Pass 1 extraction on all packages."""
        self._clear_manual_review()
        print(f"\n--- Pass 1: Extraction ({len(packages)} rows) ---")
        results = []
        calls_before = self.api_calls_made
        for i, pkg in enumerate(packages):
            print(f"\n[{i+1}/{len(packages)}] {pkg.filename} / {pkg.brand}")
            result = self.extract_single(pkg)
            results.append(result)

        calls_used = self.api_calls_made - calls_before
        cached = sum(1 for _ in results) - calls_used  # approximate
        print(f"\n--- Pass 1 done: {calls_used} API calls, "
              f"{self.cache_hits} cache hits ---")
        return results

    def validate_all(self, extractions: list[dict],
                     packages: list[ContextPackage]) -> list[dict]:
        """Run Pass 2 validation on all extractions."""
        print(f"\n--- Pass 2: Validation ({len(packages)} rows) ---")
        results = []
        calls_before = self.api_calls_made

        for i, (ext, pkg) in enumerate(zip(extractions, packages)):
            print(f"\n[{i+1}/{len(packages)}] {pkg.filename} / {pkg.brand}")
            validated = self.validate_single(ext, pkg)
            results.append(validated)

        calls_used = self.api_calls_made - calls_before
        print(f"\n--- Pass 2 done: {calls_used} API calls ---")
        return results


# ── Optimized Groq Extractor ─────────────────────────────────────────────────
# Per-parameter model routing: 7 simple params → 8b, 5 complex params → 70b.
# 8b failures escalate to 70b. Validation is implicit (per-param JSON is tiny
# and easy to validate structurally).

# Parameters handled by 8b (simple, binary/numeric, isolated sentences)
_SIMPLE_PARAMS = list(MICRO_PROMPTS_8B.keys())

# Parameters handled by 70b (interdependent, require reasoning)
_COMPLEX_PARAMS = [
    "step_through_phototherapy",
    "step_therapy_requirements",
    "steps_through_brands",
    "steps_through_generic",
    "reauth_requirements",
]

# All 13 output fields (excluding reasoning/estimated_access_score which are
# added by the complex prompt or post-processing)
_ALL_PARAMS = _SIMPLE_PARAMS + _COMPLEX_PARAMS


class OptimizedGroqExtractor:
    """Per-parameter model routing extractor for Groq.

    Routes 7 simple parameters to 8b and 5 complex parameters to 70b.
    On 8b validation failure, escalates to 70b.
    Maintains the same extract_single/validate_single/extract_all/validate_all
    interface as StandardExtractor for drop-in compatibility.
    """

    def __init__(self, api_key: str | None = None,
                 use_cache: bool = True,
                 call_spacing: float | None = None):
        self.use_cache = use_cache
        self.call_spacing = call_spacing if call_spacing is not None else CALL_SPACING
        self.api_calls_made = 0
        self.total_input_chars = 0
        self.total_output_chars = 0
        self.cache_hits = 0
        self.escalations = 0
        self.cache_dir = Path(CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manual_review_path = self.cache_dir.parent / "manual_review.txt"
        self._review_flags: list[str] = []
        self.provider = "groq-optimized"

        # Key pool — shared across both models
        if api_key:
            self._keys = [k.strip() for k in api_key.split(",") if k.strip()]
        else:
            self._keys = list(GROQ_API_KEYS)
        self._key_index = 0
        self._dead_keys: set[int] = set()  # indices of permanently failed keys

        self._model_8b = GROQ_VALIDATION_MODEL   # llama-3.1-8b-instant
        self._model_70b = GROQ_EXTRACTION_MODEL   # llama-3.3-70b-versatile
        self._8b_consecutive_failures = 0
        # 8b is off by default (6K TPM unusable on free tier).
        # Set GROQ_USE_8B=true for paid keys with higher TPM.
        from config import GROQ_USE_8B
        self._8b_disabled = not GROQ_USE_8B

        if self._keys:
            mode_note = "70b only" if self._8b_disabled else "8b+70b split"
            print(f"  Provider: Groq-70b-Focused ({mode_note})")
            print(f"  Model: {self._model_70b}")
        else:
            print("  WARNING: No GROQ_API_KEYS configured. Cache-only mode.")

    # ── Low-level LLM call ────────────────────────────────────────────

    def _next_live_key(self) -> int | None:
        """Advance to the next live key index. Returns None if all dead."""
        if len(self._dead_keys) >= len(self._keys):
            return None
        self._key_index = (self._key_index + 1) % len(self._keys)
        checked = 0
        while self._key_index in self._dead_keys and checked < len(self._keys):
            self._key_index = (self._key_index + 1) % len(self._keys)
            checked += 1
        if self._key_index in self._dead_keys:
            return None
        return self._key_index

    def _call_model(self, prompt: str, system: str, model: str) -> str:
        """Call Groq with smart key management.

        Key rotation strategy:
        1. Round-robin: advance to next key before every call
        2. Dead key (400/401/403/restricted): permanently removed
        3. Daily limit (RPD/TPD): permanently removed — detected via
           retry-after >= 120s or explicit "daily" in error message
        4. Per-minute limit (RPM/TPM): rotate to next key, don't kill
        5. Full cycle exhaustion: cooldown using retry-after (capped 10s)
        """
        prompt_chars = len(prompt) + (len(system) if system else 0)
        if prompt_chars > 400_000:
            raise RequestTooLarge(f"Prompt too large: {prompt_chars} chars")

        live_count = len(self._keys) - len(self._dead_keys)
        if live_count == 0:
            raise AllKeysExhausted("dead", "groq")

        if self._next_live_key() is None:
            raise AllKeysExhausted("dead", "groq")

        max_cycles = 2
        failures_this_cycle = 0

        for _ in range(max_cycles * len(self._keys)):
            live_count = len(self._keys) - len(self._dead_keys)
            if live_count == 0:
                raise AllKeysExhausted("dead", "groq")

            try:
                key = self._keys[self._key_index]
                text = _call_groq(
                    prompt, system=system, model=model,
                    api_key=key, call_spacing=self.call_spacing,
                )
                self.api_calls_made += 1
                self.total_input_chars += prompt_chars
                self.total_output_chars += len(text)
                return text
            except AllKeysExhausted:
                raise
            except Exception as e:
                error_str = str(e).lower()
                old = self._key_index

                rate_info = GroqRateLimitInfo(e)

                # --- Dead key: 400, 401, 403, restricted, banned ---
                is_dead = ("restricted" in error_str
                           or "unauthorized" in error_str
                           or "401" in error_str
                           or "403" in error_str
                           or ("400" in error_str
                               and "restricted" in error_str))
                if is_dead:
                    self._dead_keys.add(old)
                    live_count = len(self._keys) - len(self._dead_keys)
                    print(f"      [DEAD] key #{old+1} permanently removed "
                          f"({live_count} remaining)")
                    if self._next_live_key() is None:
                        raise AllKeysExhausted("dead", "groq")
                    continue

                # --- Daily limit (RPD/TPD exhausted) ---
                if rate_info.is_daily_limit:
                    self._dead_keys.add(old)
                    live_count = len(self._keys) - len(self._dead_keys)
                    print(f"      [DAILY] key #{old+1} hit daily limit, "
                          f"removed ({live_count} remaining)")
                    if self._next_live_key() is None:
                        raise AllKeysExhausted("daily", "groq")
                    continue

                # --- Rate limit (429 — TPM or RPM) ---
                is_rate = ("429" in error_str or "rate" in error_str
                           or "quota" in error_str or "413" in error_str
                           or "request too large" in error_str)
                if is_rate:
                    failures_this_cycle += 1

                    if failures_this_cycle >= live_count:
                        failures_this_cycle = 0
                        wait = rate_info.retry_after if rate_info.retry_after > 0 else 3.0
                        wait = min(wait, 10.0)
                        print(f"      [COOLDOWN] all {live_count} keys "
                              f"rate-limited, waiting {wait:.0f}s")
                        time.sleep(wait)
                    if self._next_live_key() is None:
                        raise AllKeysExhausted("rate", "groq")
                else:
                    print(f"      [ERR] key #{old+1}: {str(e)[:80]}")
                    if self._next_live_key() is None:
                        raise AllKeysExhausted("dead", "groq")
                    time.sleep(self.call_spacing)

        raise AllKeysExhausted("rate", "groq")

    # ── JSON parsing ─────────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        m = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return None

    # ── Caching ───────────────────────────────────────────────────────

    def _load_cache(self, key: str) -> dict | None:
        if not self.use_cache:
            return None
        path = self.cache_dir / key
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                self.cache_hits += 1
                return data
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def _save_cache(self, key: str, data: dict):
        # Normalize extraction output before persisting
        data = normalize_extraction(data)
        path = self.cache_dir / key
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ── Manual review flags ──────────────────────────────────────────

    def _log_manual_review(self, filename: str, brand: str, reason: str):
        self._review_flags.append(f"{filename} | {brand} | {reason}")

    def _flush_manual_review(self):
        """Write accumulated review flags to file (replaces previous content)."""
        if self._review_flags:
            with open(self.manual_review_path, "w") as f:
                f.write("\n".join(self._review_flags) + "\n")
        elif self.manual_review_path.exists():
            self.manual_review_path.unlink()

    def _clear_manual_review(self):
        """Reset review flags for a new run."""
        self._review_flags = []

    # ── Brand-specific snippet isolation ─────────────────────────────

    @staticmethod
    def _isolate_brand_section(full_text: str, brand: str,
                               max_chars: int = 24000) -> str:
        """Extract the brand-relevant section from multi-drug PDFs.

        For multi-drug PDFs (40K-70K chars), the full text contains
        criteria for many drugs. This isolates the section most relevant
        to the target brand by finding the approval criteria block that
        mentions the brand.

        Returns a focused snippet (max_chars). If the result still exceeds
        the token budget, the caller will chunk it further.
        """
        brand_lower = brand.lower()
        text_lower = full_text.lower()

        # Also search for generic name (e.g., ustekinumab for STELARA)
        generic_lower = BRAND_TO_GENERIC.get(brand, "").lower()
        search_names = [brand_lower]
        if generic_lower:
            search_names.append(generic_lower)

        # Strategy 1: Find the approval criteria block that mentions
        # the brand or its generic name.
        brand_criteria_patterns = []
        for name in search_names:
            brand_criteria_patterns.extend([
                f"coverage of {name}",
                f"{name} may be approved",
                f"{name} is considered",
            ])
        for pat in brand_criteria_patterns:
            m = re.search(pat, text_lower)
            if m:
                # Go back to find the section header
                block_start = max(0, m.start() - 500)
                block_end = min(len(full_text),
                                block_start + max_chars)
                return full_text[block_start:block_end]

        # Strategy 2: Find "all of the following" or "criteria are met"
        # blocks and check which one is near the brand name
        criteria_markers = [
            "all of the following criteria",
            "criteria are met",
            "criteria for initial approval",
            "approval criteria",
        ]
        for marker in criteria_markers:
            for m in re.finditer(re.escape(marker), text_lower):
                lookback = text_lower[max(0, m.start()-3000):m.start()]
                lookahead = text_lower[m.start():m.start()+3000]
                if any(name in lookback or name in lookahead
                       for name in search_names):
                    block_start = max(0, m.start() - 500)
                    block_end = min(len(full_text),
                                    block_start + max_chars)
                    return full_text[block_start:block_end]

        # Strategy 3: Find brand/generic name near criteria keywords
        for name in search_names:
            for m in re.finditer(re.escape(name), text_lower):
                start = max(0, m.start() - 1000)
                end = min(len(full_text), m.end() + 1000)
                context = text_lower[start:end]
                if any(kw in context for kw in [
                    "criteria", "authorization", "approval",
                    "trial", "failed", "all of the following",
                ]):
                    block_start = max(0, m.start() - 1000)
                    block_end = min(len(full_text),
                                    block_start + max_chars)
                    return full_text[block_start:block_end]

        # Strategy 4: Find [PsO Section] marker
        pso_idx = text_lower.find("[pso section")
        if pso_idx >= 0:
            return full_text[pso_idx:pso_idx + max_chars]

        # Fallback: head of text
        return full_text[:max_chars]

    # ── Parameter-specific snippet extraction ────────────────────────

    # Map simple param names to PARAMETER_PATTERNS keys
    _PARAM_TO_PATTERN_KEY = {
        "tb_test_required": "tb_test",
        "reauth_required": "reauth",
        "age": "age",
        "initial_auth_duration": "auth_duration",
        "reauth_duration": "reauth",
        "specialist_types": "specialist",
        "quantity_limits": "quantity_limit",
    }

    def _extract_snippet_for_param(self, param: str, full_text: str,
                                   brand: str) -> str:
        """Extract parameter-relevant sentences from full text.

        Uses token-budget formula to determine max snippet size for 8b,
        then fills with keyword-matched sentences up to that limit.
        """
        # Compute budget for a micro-prompt on 8b
        import prompts as P
        template = P.MICRO_PROMPTS_8B.get(param)
        if template:
            overhead = (len(template.format(brand=brand, snippet=""))
                        + len(P.MICRO_SYSTEM_8B))
        else:
            overhead = 200 + len(P.MICRO_SYSTEM_8B)
        max_chars = compute_max_context_chars(overhead, "8b")

        pattern_key = self._PARAM_TO_PATTERN_KEY.get(param)
        if not pattern_key:
            return full_text[:max_chars]

        keywords = PARAMETER_PATTERNS.get(pattern_key, [])
        if not keywords:
            return full_text[:max_chars]

        # Split into sentences (rough split on period/newline)
        sentences = re.split(r'(?<=[.!?])\s+|\n', full_text)

        # Collect sentences matching any keyword, up to budget
        matched = []
        total_chars = 0
        brand_lower = brand.lower()
        for sent in sentences:
            sent_lower = sent.lower()
            if any(kw in sent_lower for kw in keywords) or brand_lower in sent_lower:
                matched.append(sent.strip())
                total_chars += len(sent)
                if total_chars >= max_chars:
                    break

        if matched:
            return "\n".join(matched)

        # No matches — return head of text up to budget
        return full_text[:max_chars]

    # ── Simple parameter extraction (8b) ──────────────────────────────

    def _extract_simple_param(self, param: str, brand: str,
                              snippet: str) -> dict | None:
        """Extract one simple parameter via 8b. Returns single-key dict or None."""
        prompt = build_micro_prompt(param, brand, snippet)
        response = self._call_model(prompt, MICRO_SYSTEM_8B, self._model_8b)
        result = self._parse_json(response)
        if result and param in result:
            return result
        return None

    def _extract_simple_param_escalated(self, param: str, brand: str,
                                        snippet: str) -> dict | None:
        """Retry a simple parameter on 70b after 8b validation failure."""
        prompt = build_micro_prompt(param, brand, snippet)
        response = self._call_model(prompt, MICRO_SYSTEM_8B, self._model_70b)
        result = self._parse_json(response)
        if result and param in result:
            self.escalations += 1
            return result
        return None

    # ── Validation helpers ────────────────────────────────────────────

    @staticmethod
    def _validate_simple_value(param: str, value) -> bool:
        """Structural validation for simple parameter values."""
        s = str(value).strip()
        if not s:
            return False

        if param == "tb_test_required":
            return s in ("Y", "No", "NA")
        if param == "reauth_required":
            return s in ("Yes", "No", "NA")
        if param == "age":
            # Accept: "No", "NA", "FDA labelled age", ">=18", ">=6", etc.
            return bool(s)
        if param in ("initial_auth_duration", "reauth_duration"):
            return s.isdigit() or s in ("NA", "Unspecified", "No")
        if param == "specialist_types":
            return bool(s)
        if param == "quantity_limits":
            return bool(s)
        return True

    # ── Complex parameter extraction ─────────────────────────────────

    def _extract_complex_with_preanalysis(self, package: ContextPackage,
                                          focused_text: str
                                          ) -> tuple[dict | None, str]:
        """Extract 5 complex params using pre-analysis + 8b confirmation.

        Args:
            package: Context package with metadata
            focused_text: Brand-isolated snippet (not full 73K text)

        Returns (result_dict, model_used).
        Falls back to 70b direct if 8b confirmation fails.
        """
        brand = package.brand
        text = focused_text
        universal = package.universal_criteria_text or ""

        # Step 1: Deterministic pre-analysis
        analysis = analyze_step_therapy(text, brand, universal)
        pre_block = analysis.to_prompt_block()

        # Step 2: Try 8b confirmation (skipped if auto-disabled)
        prompt = build_confirmation_prompt(brand, pre_block, text)
        response = self._try_8b(prompt, CONFIRMATION_SYSTEM)
        if response:
            result = self._parse_json(response)
            if result:
                required = {"step_through_phototherapy", "steps_through_brands",
                            "steps_through_generic"}
                if required.issubset(result.keys()):
                    return result, f"8b-confirmed ({analysis.confidence})"

        # Step 3: 8b failed/disabled — try 70b with confirmation prompt
        # (pre-analysis still helps 70b be more accurate)
        response = self._call_model(prompt, CONFIRMATION_SYSTEM,
                                    self._model_70b)
        result = self._parse_json(response)
        if result:
            required = {"step_through_phototherapy", "steps_through_brands",
                        "steps_through_generic"}
            if required.issubset(result.keys()):
                return result, "70b-confirmed"

        # Step 4: Both failed — try 70b direct (no pre-analysis)
        try:
            prompt = build_complex_prompt(
                brand, text, package.document_type,
                package.preferred_status)
            response = self._call_model(prompt, MICRO_SYSTEM_70B,
                                        self._model_70b)
            result = self._parse_json(response)
            if result:
                required = {"step_through_phototherapy",
                            "steps_through_brands",
                            "steps_through_generic"}
                if required.issubset(result.keys()):
                    return result, "70b-direct"
        except RequestTooLarge:
            pass

        return None, "failed"

    # ── Main extraction ───────────────────────────────────────────────

    def _try_8b(self, prompt: str, system: str) -> str:
        """Try 8b model. Returns response or empty string.

        Pre-checks:
        - Skip if 8b already disabled (previous failures)
        - Skip if prompt exceeds 8b's token budget: r * (T_8b - output_reserve)

        Auto-disables 8b on first failure — a TPM error on request #1
        means the prompt is too large or the rate limit is too tight.
        No key rotation will fix a per-model TPM limit.
        """
        if self._8b_disabled:
            return ""

        # Pre-check: skip if total prompt exceeds 8b's input token budget
        from config import (GROQ_TOKEN_BUDGET_8B, GROQ_OUTPUT_TOKENS_8B)
        max_input_tokens = GROQ_TOKEN_BUDGET_8B - GROQ_OUTPUT_TOKENS_8B
        prompt_chars = len(prompt) + (len(system) if system else 0)
        # Optimized mode uses micro-prompts (~1K overhead) — estimate at 1:1
        estimated_tokens = prompt_chars
        if estimated_tokens > max_input_tokens:
            print(f"      [8b SKIP] prompt too large "
                  f"(~{estimated_tokens:,} tokens > {max_input_tokens:,})")
            self._8b_disabled = True
            return ""

        response = self._call_model(prompt, system, self._model_8b)
        if response:
            self._8b_consecutive_failures = 0
            return response

        # First failure = disable. 8b's 6K TPM is per-org, not per-key.
        # If it fails once, rotating keys won't help.
        self._8b_consecutive_failures += 1
        self._8b_disabled = True
        print(f"      [8b DISABLED] failed on first attempt, "
              f"routing all calls to 70b")
        return ""

    def _extract_simple_batched(self, brand: str, snippet: str) -> tuple:
        """Extract all 7 simple params in one call.

        Tries 8b first (saves 70b daily budget). Falls back to 70b.
        Returns (valid_dict, model_used).
        """
        prompt = build_batched_simple_prompt(brand, snippet)

        # Try 8b first (skipped if auto-disabled)
        response = self._try_8b(prompt, MICRO_SYSTEM_8B)
        if response:
            result = self._parse_json(response)
            if result:
                valid = {}
                for param in _SIMPLE_PARAMS:
                    if param in result and self._validate_simple_value(
                            param, result[param]):
                        valid[param] = result[param]
                if len(valid) >= 5:
                    return valid, "8b"

        # 8b failed or disabled — use 70b
        response = self._call_model(prompt, MICRO_SYSTEM_8B, self._model_70b)
        result = self._parse_json(response)
        if result:
            valid = {}
            for param in _SIMPLE_PARAMS:
                if param in result and self._validate_simple_value(
                        param, result[param]):
                    valid[param] = result[param]
            if valid:
                return valid, "70b"

        return {}, "failed"

    def _extract_from_text(self, package: ContextPackage,
                           text: str, chunk_label: str = "") -> dict:
        """Extract all 13 parameters from a single text segment.

        Stage 1: 7 simple params in one batched call (8b, fallback 70b)
        Stage 2: 5 complex params via 70b
        """
        brand = package.brand
        prefix = f"    [{chunk_label}] " if chunk_label else "    "
        merged = {}

        # Stage 1: Batched simple params
        batched, model_used = self._extract_simple_batched(brand, text)
        if batched:
            merged.update(batched)
            print(f"{prefix}simple params: {len(batched)}/7 OK ({model_used})")

        # Fill missing simple params with NA
        missing = [p for p in _SIMPLE_PARAMS if p not in merged]
        if missing:
            for param in missing:
                merged[param] = "NA"
            if len(missing) > 3:
                print(f"{prefix}simple params missing ({len(missing)}) -> NA")

        # Stage 2: Complex params via pre-analysis + 70b
        complex_result, complex_model = self._extract_complex_with_preanalysis(
            package, text)
        if complex_result:
            merged.update(complex_result)
            print(f"{prefix}complex params: OK ({complex_model})")
        else:
            for param in _COMPLEX_PARAMS:
                if param not in merged:
                    merged[param] = "NA"
            merged.setdefault("reasoning", "Complex extraction failed")
            print(f"{prefix}complex params: FALLBACK to NA")

        return merged

    @staticmethod
    def _merge_chunk_results(results: list[dict]) -> dict:
        """Merge extractions from multiple chunks with field-aware logic."""
        return _merge_results_field_aware(results)

    def _compute_chunk_budget(self) -> int:
        """Compute max context chars per chunk for the optimized extractor.

        Each chunk is sent to both the simple-batch prompt (8b) and the
        complex prompt (70b). The chunk budget is the tighter of the two
        so neither call exceeds its model's token limit.

        The builder functions (build_batched_simple_prompt, build_complex_prompt)
        also truncate internally using the same formula, so this budget
        matches what the LLM actually receives.
        """
        import prompts as P
        simple_overhead = (len(P.BATCHED_SIMPLE_PROMPT.format(brand="X", snippet=""))
                           + len(P.MICRO_SYSTEM_8B))
        complex_overhead = (len(P.COMPLEX_PARAMS_PROMPT.format(
            brand="X", snippet="", doc_type="multi-drug", preferred="Non-preferred"))
            + len(P.MICRO_SYSTEM_70B))

        budget_simple = compute_max_context_chars(simple_overhead, "8b")
        budget_complex = compute_max_context_chars(complex_overhead, "70b")

        return min(budget_simple, budget_complex)

    def extract_single(self, package: ContextPackage) -> dict:
        """Extract all 13 parameters using split-prompt strategy.

        For large texts, splits into chunks sized to fit within the
        token budget (r*T - P), extracts from each, and merges.
        """
        cache_key = _cache_key(package.filename, package.brand, "extract", self.provider)
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        brand = package.brand
        full_text = package.full_relevant_text

        # Isolate brand-specific section for multi-drug PDFs
        if len(full_text) > 10_000:
            focused_text = self._isolate_brand_section(full_text, brand)
        else:
            focused_text = full_text

        # Compute token-budget-aware chunk size
        chunk_budget = self._compute_chunk_budget()

        # Check if chunking is needed
        if len(focused_text) > chunk_budget:
            # Split into chunks and extract from each
            chunks = StandardExtractor._split_into_chunks(
                focused_text, chunk_budget)

            # Guarantee no chunk exceeds the limit (hard-split fallback)
            safe_chunks = []
            for c in chunks:
                if len(c) <= chunk_budget:
                    safe_chunks.append(c)
                else:
                    for i in range(0, len(c), chunk_budget):
                        safe_chunks.append(c[i:i + chunk_budget])
            chunks = safe_chunks

            print(f"  [{brand}] {package.filename} -> chunked LLM analysis "
                  f"({len(full_text):,} -> {len(focused_text):,} chars, "
                  f"{len(chunks)} chunks, budget {chunk_budget:,} chars/chunk)")

            chunk_results = []
            for i, chunk in enumerate(chunks):
                result = self._extract_from_text(
                    package, chunk, chunk_label=f"chunk {i+1}/{len(chunks)}")
                chunk_results.append(result)

            merged = self._merge_chunk_results(chunk_results)
        else:
            size_info = (f"{len(full_text):,} -> {len(focused_text):,} chars"
                         if len(focused_text) != len(full_text)
                         else f"{len(full_text):,} chars")
            print(f"  [{brand}] {package.filename} -> LLM analysis "
                  f"({size_info})")
            merged = self._extract_from_text(package, focused_text)

        # Normalize before fallback check (so _count_real_params sees clean values)
        merged = normalize_extraction(merged)

        # Raw-PDF fallback: if too few real params and raw text is small
        # Note: normalize_extraction also runs inside _save_cache, but we
        # need it here so the fallback check sees clean values.
        # enough to send directly, re-extract from unfiltered text.
        from scorer import _count_real_params
        real = _count_real_params(merged)
        # Raw fallback must fit in a single call to both 8b and 70b,
        # so use the chunk budget (min of both models' capacity).
        raw_fallback_max = self._compute_chunk_budget()
        if real < 5 and len(full_text) <= raw_fallback_max and focused_text != full_text:
            print(f"  [{brand}] FALLBACK: only {real} real params, "
                  f"re-extracting from raw PDF ({len(full_text):,} chars)")
            fallback = normalize_extraction(
                self._extract_from_text(package, full_text,
                                        chunk_label="raw-fallback"))
            # Merge: fallback fills NA slots only, never overwrites
            na_values = {"na", "n/a", "not applicable", "not mentioned", ""}
            for k, v in fallback.items():
                existing = str(merged.get(k, "NA")).strip().lower()
                if existing in na_values and str(v).strip().lower() not in na_values:
                    merged[k] = v

        merged.setdefault("estimated_access_score", 50)
        merged.setdefault("reasoning", "")

        self._save_cache(cache_key, merged)
        return merged

    def validate_single(self, extraction: dict,
                        package: ContextPackage) -> dict:
        """Validate extraction. For groq-optimized, per-param extraction
        already produces focused results, so validation is lighter —
        just cross-check step counts for consistency."""
        cache_key = _cache_key(package.filename, package.brand, "validate", self.provider)
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        # Light structural validation — no extra API call needed for most rows
        result = dict(extraction)
        corrections = []

        # Cross-check: if reauth_duration or reauth_requirements is non-NA,
        # reauth_required should be "Yes"
        reauth_dur = str(result.get("reauth_duration", "NA")).strip()
        reauth_req_text = str(result.get("reauth_requirements", "NA")).strip()
        reauth_flag = str(result.get("reauth_required", "NA")).strip()

        if (reauth_dur not in ("NA", "N/A", "") or
                reauth_req_text not in ("NA", "N/A", "")):
            if reauth_flag != "Yes":
                result["reauth_required"] = "Yes"
                corrections.append("Set reauth_required=Yes (duration/requirements present)")

        if reauth_flag == "Yes" and reauth_dur in ("NA", "N/A", ""):
            result["reauth_duration"] = "Unspecified"
            corrections.append("Set reauth_duration=Unspecified (reauth required but no duration)")

        if corrections:
            result["corrections"] = corrections
            print(f"  [{package.brand}] Validated ({len(corrections)} correction(s))")
        else:
            result["corrections"] = None
            print(f"  [{package.brand}] Validated (no changes)")

        self._save_cache(cache_key, result)
        return result

    # ── Batch operations ──────────────────────────────────────────────

    def extract_all(self, packages: list[ContextPackage]) -> list[dict]:
        """Run extraction on all packages."""
        self._clear_manual_review()
        print(f"\n--- Optimized Extraction ({len(packages)} rows) ---")
        results = []
        calls_before = self.api_calls_made
        for i, pkg in enumerate(packages):
            print(f"\n[{i+1}/{len(packages)}] {pkg.filename} / {pkg.brand}")
            result = self.extract_single(pkg)
            results.append(result)

        calls_used = self.api_calls_made - calls_before
        print(f"\n--- Extraction done: {calls_used} API calls, "
              f"{self.cache_hits} cache hits, "
              f"{self.escalations} escalations ---")
        return results

    def validate_all(self, extractions: list[dict],
                     packages: list[ContextPackage]) -> list[dict]:
        """Run validation on all extractions."""
        print(f"\n--- Validation ({len(packages)} rows) ---")
        results = []
        calls_before = self.api_calls_made

        for i, (ext, pkg) in enumerate(zip(extractions, packages)):
            print(f"\n[{i+1}/{len(packages)}] {pkg.filename} / {pkg.brand}")
            validated = self.validate_single(ext, pkg)
            results.append(validated)

        calls_used = self.api_calls_made - calls_before
        print(f"\n--- Validation done: {calls_used} API calls ---")
        return results

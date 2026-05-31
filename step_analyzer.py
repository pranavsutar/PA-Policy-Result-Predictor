"""
step_analyzer.py — Deterministic pre-analysis of step therapy requirements.

Parses policy text to extract step therapy structure (AND/OR blocks, drug
mentions, step counts) BEFORE sending to the LLM. The LLM then confirms
or corrects the pre-analysis instead of reasoning from scratch.

This enables accurate extraction with 8b models by reducing the task from
multi-hop reasoning to simple confirmation.
"""
import re
from dataclasses import dataclass, field

from config import BIOLOGIC_BRANDS, BIOSIMILAR_BRANDS, GENERIC_DRUGS, BRAND_TO_GENERIC


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class DrugMention:
    name: str
    classification: str  # "branded" or "generic"
    context: str         # surrounding sentence


@dataclass
class StepRequirement:
    count: int
    drug_type: str       # "branded", "generic", or "mixed"
    drugs: list[str]
    connector: str       # "AND" or "OR"
    source_text: str


@dataclass
class StepAnalysis:
    """Pre-digested step therapy analysis for LLM confirmation."""
    branded_steps: int | str       # integer count or "NA"
    generic_steps: int | str       # integer count or "NA"
    phototherapy_mandatory: bool | None  # True/False/None(unknown)
    phototherapy_in_or: bool
    requirements: list[StepRequirement]
    drug_mentions: list[DrugMention]
    reauth_text: str               # extracted reauth-relevant sentences
    confidence: str                # "high", "medium", "low"
    reasoning: str                 # explanation of how counts were derived

    def to_prompt_block(self) -> str:
        """Format as a text block for the LLM confirmation prompt."""
        lines = ["PRE-ANALYSIS RESULTS:"]

        if self.requirements:
            lines.append(f"  Detected {len(self.requirements)} step requirement(s):")
            for i, req in enumerate(self.requirements, 1):
                drugs_str = ", ".join(req.drugs[:5])
                if len(req.drugs) > 5:
                    drugs_str += f" (+{len(req.drugs)-5} more)"
                lines.append(
                    f"    {i}. {req.count} {req.drug_type} step(s) "
                    f"[{req.connector}]: {drugs_str}"
                )
                if req.source_text:
                    # Truncate source text for prompt budget
                    src = req.source_text[:200].replace("\n", " ")
                    lines.append(f"       Source: \"{src}\"")
        else:
            lines.append("  No explicit step requirements detected.")

        # Phototherapy
        if self.phototherapy_mandatory is True:
            lines.append("  Phototherapy: MANDATORY (standalone requirement, not in OR)")
        elif self.phototherapy_in_or:
            lines.append("  Phototherapy: mentioned but in OR with other options (not mandatory)")
        elif self.phototherapy_mandatory is False:
            lines.append("  Phototherapy: not required")
        else:
            lines.append("  Phototherapy: not mentioned")

        # Computed counts
        lines.append("")
        lines.append("COMPUTED VALUES:")
        lines.append(f"  steps_through_brands: {self.branded_steps}")
        lines.append(f"  steps_through_generic: {self.generic_steps}")
        if self.phototherapy_mandatory is True:
            lines.append('  step_through_phototherapy: "Yes"')
        elif self.phototherapy_mandatory is False or self.phototherapy_in_or:
            lines.append('  step_through_phototherapy: "No"')
        else:
            lines.append('  step_through_phototherapy: "N/A"')

        lines.append(f"  Analysis confidence: {self.confidence}")
        if self.reasoning:
            lines.append(f"  Reasoning: {self.reasoning}")

        return "\n".join(lines)


# ── Drug mention extraction ─────────────────────────────────────────────────

# Build lookup sets (lowercase for matching)
_BRANDED_LOWER = {b.lower() for b in BIOLOGIC_BRANDS}  # originator biologics only
_BIOSIMILAR_LOWER = {b.lower() for b in BIOSIMILAR_BRANDS}
_GENERIC_LOWER = {g.lower() for g in GENERIC_DRUGS} | _BIOSIMILAR_LOWER

# Generic names of originator biologics count as branded
_GENERIC_NAMES = {v.lower() for k, v in BRAND_TO_GENERIC.items()
                  if k.lower() not in BIOSIMILAR_BRANDS}
_BRANDED_LOWER.update(_GENERIC_NAMES)


def _find_drug_mentions(text: str) -> list[DrugMention]:
    """Scan text for all drug name mentions with classification."""
    mentions = []
    text_lower = text.lower()

    # Build a combined pattern for all drug names
    all_drugs = {}
    for drug in BIOLOGIC_BRANDS:
        all_drugs[drug.lower()] = "branded"
    for drug in BIOSIMILAR_BRANDS:
        all_drugs[drug.lower()] = "generic"  # biosimilars = generic steps
    for drug in GENERIC_DRUGS:
        all_drugs[drug.lower()] = "generic"
    # Add generic names of biologics — classify based on whether
    # the brand is an originator or biosimilar
    for brand, generic in BRAND_TO_GENERIC.items():
        if brand.lower() in BIOSIMILAR_BRANDS:
            all_drugs[generic.lower()] = "generic"
        else:
            all_drugs[generic.lower()] = "branded"

    for drug_name, classification in all_drugs.items():
        # Skip very short names that cause false positives
        if len(drug_name) < 4:
            continue
        for m in re.finditer(r'\b' + re.escape(drug_name) + r'\b', text_lower):
            # Get surrounding context (±100 chars)
            start = max(0, m.start() - 100)
            end = min(len(text), m.end() + 100)
            context = text[start:end].replace("\n", " ").strip()
            mentions.append(DrugMention(
                name=drug_name,
                classification=classification,
                context=context,
            ))

    return mentions


# ── Step therapy pattern detection ───────────────────────────────────────────

# Patterns for detecting step counts
_STEP_PATTERNS = [
    # "trial of THREE preferred products"
    (r'(?i)(?:trial|failure|failed|inadequate\s+response)\s+(?:of|to)\s+'
     r'(ONE|TWO|THREE|FOUR|FIVE|\d+)\s+'
     r'(?:preferred\s+)?(?:product|agent|biologic|medication)s?',
     "branded"),

    # "at least N preferred biologic products"
    (r'(?i)(?:at\s+least\s+|minimum\s+of\s+)?'
     r'(one|two|three|four|five|\d+)\s*(?:\(\d+\)\s*)?'
     r'preferred\s+biologic\s+(?:product|agent|medication)s?',
     "branded"),

    # "at least N systemic therapy/therapies"
    (r'(?i)(?:at\s+least\s+|minimum\s+of\s+)?'
     r'(one|two|three|four|five|\d+)\s*(?:\(\d+\)\s*)?'
     r'(?:systemic|conventional|non-biologic)\s+'
     r'(?:therap(?:y|ies)|agent|medication|DMARD)s?',
     "generic"),

    # "ONE formulary conventional prerequisite agent"
    (r'(?i)(ONE|TWO|THREE|\d+)\s+'
     r'(?:formulary\s+)?conventional\s+prerequisite\s+agent',
     "generic"),

    # "previously received a biologic" (= 1 branded step)
    (r'(?i)previously\s+received\s+(?:a|one)\s+'
     r'(?:biologic|targeted\s+(?:immune|synthetic)\s+(?:drug|modulator))',
     "branded_1"),

    # "unable to take N preferred products"
    (r'(?i)unable\s+to\s+take\s+'
     r'(ONE|TWO|THREE|FOUR|FIVE|\d+)\s+'
     r'preferred\s+products?',
     "branded"),

    # "trial and failure of N topical/conventional therapies"
    (r'(?i)(?:trial|failure)\s+(?:of|and\s+failure\s+(?:of|to))\s+'
     r'(?:at\s+least\s+)?'
     r'(one|two|three|\d+)\s*(?:\(\d+\)\s*)?'
     r'(?:of\s+the\s+following\s+)?'
     r'(?:topical|conventional)\s+therap(?:y|ies)',
     "generic"),

    # "trial and failure ... methotrexate" (specific generic drug)
    (r'(?i)(?:trial|failure|inadequate)\s+(?:of|to|and\s+failure\s+(?:of|to))\s+'
     r'.*?(?:methotrexate|cyclosporine|acitretin)',
     "generic_mention"),

    # "trial and failure ... to one of the following topical therapies"
    (r'(?i)(?:trial\s+and\s+failure|trial|failure|intolerance)\s*'
     r'(?:,\s*(?:contraindication|intolerance)\s*)*'
     r'(?:,?\s*or\s+(?:contraindication|intolerance)\s+)?'
     r'to\s+(?:one|any)\s+of\s+the\s+following\s+'
     r'(?:topical\s+)?(?:therap(?:y|ies)|medication|agent)',
     "generic_mention"),

    # "inadequate response to ... one (1) systemic therapy (e.g., methotrexate)"
    (r'(?i)(?:inadequate\s+response|therapeutic\s+failure|intolerance)\s+'
     r'to\s+(?:at\s+least\s+)?'
     r'(one|two|three|\d+)\s*(?:\(\d+\)\s*)?'
     r'(?:of\s+the\s+following\s+)?'
     r'(?:systemic|topical|conventional)\s+'
     r'(?:therap(?:y|ies)|agent|medication)',
     "generic"),

    # "trial and failure of ONE preferred anti-TNF" or "ONE preferred agent"
    (r'(?i)(?:trial|failure|require)\s+(?:of|and\s+failure\s+(?:of|to))?\s*'
     r'(ONE|TWO|THREE|FOUR|FIVE|\d+)\s+'
     r'preferred\s+(?:anti-TNF|anti-tumor|agent|product)',
     "branded"),

    # "require trial and failure of ALL preferred agents"
    (r'(?i)require\s+(?:trial\s+and\s+failure\s+of\s+)?'
     r'ALL\s+preferred\s+(?:agent|product)s?',
     "branded_all"),

    # "previously received a biologic or targeted synthetic drug"
    (r'(?i)previously\s+received\s+a\s+'
     r'(?:biologic|targeted\s+synthetic\s+drug)',
     "branded_1"),
]

# Number word to int
_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
}


def _parse_count(s: str) -> int:
    """Parse a number word or digit string to int."""
    return _NUM_WORDS.get(s.lower().strip(), 1)


# ── OR/AND connector detection ───────────────────────────────────────────────

def _detect_connector(text: str) -> str:
    """Detect the dominant logical connector in a text block.

    Returns "AND" or "OR".
    """
    text_lower = text.lower()

    # Explicit markers
    and_markers = [
        "all of the following",
        "each of the following",
        "must meet all",
        "a. through",  # lettered list with "through" = AND
    ]
    or_markers = [
        "one of the following",
        "any of the following",
        "either of the following",
        "one (1) of",
        "any (1) of",
        "meets one",
        "meets any",
    ]

    for marker in and_markers:
        if marker in text_lower:
            return "AND"
    for marker in or_markers:
        if marker in text_lower:
            return "OR"

    # Count explicit connectors
    or_count = len(re.findall(r'\bor\b', text_lower))
    and_count = len(re.findall(r'\band\b', text_lower))

    # Per business rules: no connector between statements = OR
    if or_count > and_count:
        return "OR"
    if and_count > 0:
        return "AND"
    return "OR"  # default per business rules


# ── Phototherapy analysis ────────────────────────────────────────────────────

def _analyze_phototherapy(text: str) -> tuple[bool | None, bool]:
    """Analyze phototherapy requirements.

    Returns (is_mandatory, is_in_or).
    - is_mandatory: True if phototherapy is a standalone AND requirement
    - is_in_or: True if phototherapy appears in an OR block
    """
    text_lower = text.lower()

    photo_keywords = ["phototherapy", "puva", "uvb", "psoralen", "light therapy"]
    has_phototherapy = any(kw in text_lower for kw in photo_keywords)

    if not has_phototherapy:
        return None, False

    # Find sentences containing phototherapy
    sentences = re.split(r'(?<=[.!?])\s+|\n', text)
    photo_sentences = [s for s in sentences
                       if any(kw in s.lower() for kw in photo_keywords)]

    is_in_or = False
    is_mandatory = False

    for sent in photo_sentences:
        sent_lower = sent.lower()

        # Check if phototherapy is in an OR with other options
        # Pattern: "phototherapy OR methotrexate" or "methotrexate or phototherapy"
        if re.search(r'(?i)(?:phototherapy|puva|uvb).*?\bor\b', sent_lower):
            is_in_or = True
        if re.search(r'(?i)\bor\b.*?(?:phototherapy|puva|uvb)', sent_lower):
            is_in_or = True

        # Check for "one of the following" containing phototherapy
        if any(m in sent_lower for m in ["one of the following", "any of the following",
                                          "either"]):
            is_in_or = True

        # Standalone mandatory: "has been ineffective" or "trial of phototherapy"
        # without OR context
        if not is_in_or:
            if re.search(r'(?i)(?:phototherapy|puva|uvb)\s+(?:has been|was)\s+'
                         r'(?:ineffective|inadequate|unsuccessful)', sent_lower):
                is_mandatory = True
            if re.search(r'(?i)(?:trial|failure)\s+(?:of|and)\s+'
                         r'(?:phototherapy|puva|uvb)', sent_lower):
                # Only mandatory if not in an OR block
                is_mandatory = True

    # If in OR, it's not mandatory
    if is_in_or:
        is_mandatory = False

    return is_mandatory, is_in_or


# ── Reauthorization text extraction ──────────────────────────────────────────

def _extract_reauth_text(text: str) -> str:
    """Extract reauthorization-relevant sentences."""
    reauth_keywords = [
        "reauthorization", "renewal", "continuation criteria",
        "continuation of therapy", "re-authorization",
        "continued approval", "renewal criteria",
        "reauthorization requirements",
    ]

    sentences = re.split(r'(?<=[.!?])\s+|\n', text)
    matched = []
    total = 0
    for sent in sentences:
        if any(kw in sent.lower() for kw in reauth_keywords):
            matched.append(sent.strip())
            total += len(sent)
            if total > 1500:
                break

    return "\n".join(matched) if matched else ""


# ── Main analysis function ───────────────────────────────────────────────────

def analyze_step_therapy(text: str, brand: str,
                         universal_text: str = "") -> StepAnalysis:
    """Perform deterministic pre-analysis of step therapy requirements.

    Args:
        text: Full relevant text (PsO section + universal)
        brand: Target brand name
        universal_text: Universal criteria text (if separated)

    Returns:
        StepAnalysis with pre-computed step counts and confidence level.
    """
    # Phase 1: Find drug mentions
    drug_mentions = _find_drug_mentions(text)

    # Phase 2: Detect step requirements via regex
    requirements = []
    branded_total = 0
    generic_total = 0
    confidence = "high"
    reasoning_parts = []

    # Universal criteria are skipped for regex analysis. They contain
    # criteria for multiple drug classes (TNFs, CAM antagonists, etc.)
    # and regex cannot reliably determine which class applies to the
    # target brand. The LLM handles universal criteria interpretation
    # using the confirmation prompt.
    if universal_text:
        reasoning_parts.append(
            f"Universal text present ({len(universal_text)} chars) "
            f"— deferred to LLM")

    # Analyze PsO-specific text (full text minus universal)
    pso_text = text
    if universal_text and text.startswith(universal_text[:100]):
        pso_text = text[len(universal_text):]

    # Track OR paths for least-restrictive resolution
    or_branded = []
    or_generic = []
    in_or_block = False

    for pattern, drug_type in _STEP_PATTERNS:
        for m in re.finditer(pattern, pso_text):
            if drug_type == "branded_1":
                count = 1
                dtype = "branded"
            elif drug_type == "branded_all":
                count = 2  # conservative default for "ALL"
                dtype = "branded"
            elif drug_type == "generic_mention":
                count = 1
                dtype = "generic"
            else:
                count = _parse_count(m.group(1)) if m.lastindex else 1
                dtype = drug_type

            start = max(0, m.start() - 50)
            end = min(len(pso_text), m.end() + 100)
            source = pso_text[start:end]

            # Check if this is inside an OR block
            wider_ctx = pso_text[max(0, m.start()-300):m.end()+300]
            connector = _detect_connector(wider_ctx)

            ctx_mentions = _find_drug_mentions(source)
            drug_names = [d.name for d in ctx_mentions]

            req = StepRequirement(
                count=count, drug_type=dtype, drugs=drug_names,
                connector=connector, source_text=source,
            )
            requirements.append(req)

            if connector == "OR":
                if dtype == "branded":
                    or_branded.append(count)
                else:
                    or_generic.append(count)
            else:
                if dtype == "branded":
                    branded_total += count
                elif dtype == "generic":
                    generic_total += count

            reasoning_parts.append(
                f"PsO [{connector}]: {count} {dtype} "
                f"(from: {source[:80]})")
            break  # take first match per pattern

    # Resolve OR paths — take least restrictive
    if or_branded and or_generic:
        # OR between branded and generic paths — take generic (fewer barriers)
        least = min(or_generic)
        generic_total += least
        reasoning_parts.append(
            f"OR resolution: branded paths {or_branded} vs "
            f"generic paths {or_generic} -> least restrictive = "
            f"{least} generic")
    elif or_branded:
        branded_total += min(or_branded)
        reasoning_parts.append(
            f"OR resolution: {min(or_branded)} branded (least of {or_branded})")
    elif or_generic:
        generic_total += min(or_generic)
        reasoning_parts.append(
            f"OR resolution: {min(or_generic)} generic (least of {or_generic})")

    # Phase 3: Phototherapy
    photo_mandatory, photo_in_or = _analyze_phototherapy(text)

    # Phase 4: Reauth text
    reauth_text = _extract_reauth_text(text)

    # Phase 5: Confidence assessment
    if not requirements and not drug_mentions:
        confidence = "low"
        reasoning_parts.append("No step patterns or drug mentions found")
    elif not requirements and drug_mentions:
        confidence = "medium"
        reasoning_parts.append(
            f"No step patterns matched but found {len(drug_mentions)} "
            f"drug mentions — LLM should determine structure")
    elif len(requirements) >= 2:
        confidence = "high"
    else:
        confidence = "medium"

    # Final counts
    final_branded = branded_total if branded_total > 0 else "NA"
    final_generic = generic_total if generic_total > 0 else "NA"

    return StepAnalysis(
        branded_steps=final_branded,
        generic_steps=final_generic,
        phototherapy_mandatory=photo_mandatory,
        phototherapy_in_or=photo_in_or,
        requirements=requirements,
        drug_mentions=drug_mentions,
        reauth_text=reauth_text,
        confidence=confidence,
        reasoning=" | ".join(reasoning_parts) if reasoning_parts else "",
    )

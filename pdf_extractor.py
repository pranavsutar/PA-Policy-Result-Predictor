"""
pdf_extractor.py — Stage 1 (PDF text extraction + page indexing) and
                    Stage 2 (section localization + structuring).

Extracts text from PDFs, identifies relevant pages, detects universal criteria
sections, and builds structured context packages for each (PDF, Brand) pair.
"""
import os
import re
from dataclasses import dataclass, field

import fitz  # PyMuPDF

from config import (
    PRIMARY_KEYWORDS, BRAND_KEYWORDS, SECONDARY_KEYWORDS,
    UNIVERSAL_SECTION_KEYWORDS, BRAND_TO_GENERIC, get_pdf_dir,
    FILTER_LEVEL, PARAMETER_PATTERNS,
    NON_PSO_INDICATION_KEYWORDS, has_pso_signal,
)


@dataclass
class PageInfo:
    page_num: int  # 0-indexed
    text: str
    tag: str  # "relevant", "context", "preamble", "universal", "irrelevant"
    relevance_score: float = 0.0


@dataclass
class ContextPackage:
    """Structured context for one (PDF, Brand) extraction."""
    filename: str
    brand: str
    universal_criteria_text: str = ""
    psoriasis_section_text: str = ""
    brand_specific_text: str = ""
    reauthorization_text: str = ""
    quantity_limit_text: str = ""
    preferred_status: str = "unknown"  # "preferred", "non-preferred", "unknown"
    document_type: str = "unknown"  # "single-drug", "multi-drug", "flat-catalog"
    total_pages: int = 0
    relevant_pages_used: int = 0
    full_relevant_text: str = ""  # concatenated text sent to LLM


def _get_strikethrough_spans(page: fitz.Page) -> list[tuple[float, float, float, float]]:
    """Get bounding boxes of strike-through annotations on a page.

    Checks both explicit StrikeOut annotations and inline text with
    strike-through font flags.
    """
    rects = []
    # Check annotations (explicit StrikeOut type)
    for annot in page.annots() or []:
        if annot.type[0] == fitz.PDF_ANNOT_STRIKE_OUT:
            rects.append(annot.rect)
    return rects


def _remove_strikethrough_text(page: fitz.Page, page_text: str) -> str:
    """Remove text that falls within strike-through annotation regions.

    Extracts text blocks with position info, checks each against
    strike-through rectangles, and rebuilds the text without struck-through
    portions.
    """
    strike_rects = _get_strikethrough_spans(page)
    if not strike_rects:
        return page_text

    # Get text with position info (dict mode gives word-level bounding boxes)
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    clean_parts = []

    for block in blocks:
        if block["type"] != 0:  # skip image blocks
            continue
        for line in block["lines"]:
            line_parts = []
            for span in line["spans"]:
                span_rect = fitz.Rect(span["bbox"])
                # Check if this span overlaps any strike-through rect
                is_struck = False
                for sr in strike_rects:
                    if span_rect.intersects(fitz.Rect(sr)):
                        is_struck = True
                        break
                if not is_struck:
                    line_parts.append(span["text"])
            if line_parts:
                clean_parts.append("".join(line_parts))

    return "\n".join(clean_parts) if clean_parts else page_text


def extract_pdf_text(pdf_path: str) -> list[str]:
    """Extract text from each page of a PDF using PyMuPDF.

    Handles encrypted PDFs, removes strike-through annotation text,
    and provides better layout preservation than PyPDF2.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  WARNING: Cannot open PDF {pdf_path}: {e}")
        return []

    pages = []
    has_strikethrough = False

    for page in doc:
        text = page.get_text("text") or ""

        # Check for and remove strike-through text
        cleaned = _remove_strikethrough_text(page, text)
        if cleaned != text:
            has_strikethrough = True
        pages.append(cleaned)

    if has_strikethrough:
        print(f"  [StrikeOut] Removed strike-through text from {os.path.basename(pdf_path)}")

    doc.close()
    return pages


def score_page_relevance(text: str, target_brand: str) -> tuple[float, set[str]]:
    """
    Score a page's relevance to psoriasis + target brand extraction.
    Returns (score, set of matched keyword categories).
    """
    lower = text.lower()
    score = 0.0
    matched_categories = set()

    # Primary keywords (disease)
    for kw in PRIMARY_KEYWORDS:
        if kw in lower:
            score += 3.0
            matched_categories.add("disease")
            break

    # Target brand name and generic name
    brand_lower = target_brand.lower()
    generic = BRAND_TO_GENERIC.get(target_brand, "").lower()
    if brand_lower in lower or generic in lower:
        score += 5.0
        matched_categories.add("target_brand")

    # Other brand keywords (indicates multi-drug section)
    for kw in BRAND_KEYWORDS:
        if kw in lower and kw != brand_lower and kw != generic:
            score += 0.5
            matched_categories.add("other_brands")
            break

    # Secondary keywords
    for kw in SECONDARY_KEYWORDS:
        if kw in lower:
            score += 1.0
            matched_categories.add("secondary")
            break

    # Universal section keywords
    for kw in UNIVERSAL_SECTION_KEYWORDS:
        if kw in lower:
            score += 2.0
            matched_categories.add("universal")
            break

    # Reauthorization-specific
    reauth_kws = ["reauthorization", "renewal", "continuation criteria",
                  "continuation of therapy", "renewal criteria"]
    for kw in reauth_kws:
        if kw in lower:
            score += 1.5
            matched_categories.add("reauthorization")
            break

    # Quantity limit specific
    if "quantity limit" in lower or "quantity level limit" in lower:
        score += 2.0
        matched_categories.add("quantity_limit")

    return score, matched_categories


def classify_document(pages: list[str]) -> str:
    """
    Classify document structure type.
    Returns: "single-drug", "multi-drug", "flat-catalog", or "decision-tree"
    """
    n = len(pages)

    # Content-based single-drug detection: count distinct brand names
    # across the entire document. If only 0-1 brands appear, it's
    # single-drug regardless of page count.
    all_text_lower = "\n".join(pages).lower()
    distinct_brands = set()
    for kw in BRAND_KEYWORDS:
        if kw in all_text_lower:
            distinct_brands.add(kw)
    # Collapse brand/generic pairs into one entity
    _brand_generic_pairs = set()
    for brand_name, generic_name in BRAND_TO_GENERIC.items():
        _brand_generic_pairs.add((brand_name.lower(), generic_name.lower()))
    collapsed = set()
    for b in distinct_brands:
        paired = False
        for brand_name, generic_name in _brand_generic_pairs:
            if b == brand_name or b == generic_name:
                collapsed.add(brand_name)
                paired = True
                break
        if not paired:
            collapsed.add(b)

    if len(collapsed) <= 1:
        return "single-drug"

    # Short docs with few brands are still single-drug
    if n <= 10:
        return "single-drug"

    # Check for flat catalog pattern: many pages each starting with "Products Affected"
    # or each page being a self-contained drug entry
    flat_indicators = 0
    for i in range(min(10, n)):
        lower = pages[i].lower()
        if "products affected" in lower and "pa criteria" in lower:
            flat_indicators += 1

    if flat_indicators >= 3 and n > 50:
        return "flat-catalog"

    # Check for decision-tree pattern: "Go to #N" / "Yes: ... No: ..."
    dt_count = 0
    for i in range(min(30, n)):
        lower = pages[i].lower()
        if re.search(r'go\s+to\s+#?\s*\d+', lower):
            dt_count += 1
    if dt_count >= 5:
        return "decision-tree"

    return "multi-drug"


def detect_preferred_status(pages: list[str], target_brand: str) -> str:
    """
    Detect if the target brand is preferred or non-preferred.
    Searches the first 10 pages and any page mentioning preferred/non-preferred.
    """
    brand_lower = target_brand.lower()
    generic = BRAND_TO_GENERIC.get(target_brand, "").lower()

    for page_text in pages[:min(15, len(pages))]:
        lower = page_text.lower()

        # Look for explicit preferred/non-preferred lists.
        # Limit regex span to 200 chars to avoid false matches across
        # unrelated sections on the same page.
        if "non-preferred" in lower or "non preferred" in lower:
            np_patterns = [f"non-preferred.{{0,200}}{re.escape(brand_lower)}",
                           f"{re.escape(brand_lower)}.{{0,200}}non-preferred"]
            if generic:
                np_patterns += [f"non-preferred.{{0,200}}{re.escape(generic)}",
                                f"{re.escape(generic)}.{{0,200}}non-preferred"]
            for pattern in np_patterns:
                if re.search(pattern, lower):
                    return "non-preferred"

        if "preferred" in lower:
            pf_patterns = [f"preferred.{{0,200}}{re.escape(brand_lower)}"]
            if generic:
                pf_patterns.append(f"preferred.{{0,200}}{re.escape(generic)}")
            for pattern in pf_patterns:
                match = re.search(pattern, lower)
                if match and "non-preferred" not in lower[max(0, match.start()-15):match.start()]:
                    return "preferred"

    return "unknown"


def extract_universal_criteria(pages: list[str], doc_type: str,
                               brand: str = "") -> str:
    """Extract universal/general criteria text that applies to all brands.

    Delegates to _extract_tier1_universal which uses structural boundary
    detection and explicit header scanning.
    """
    return _extract_tier1_universal(pages, doc_type, brand)


def extract_reauthorization_text(pages: list[str], relevant_page_indices: set[int]) -> str:
    """Extract reauthorization/renewal criteria text."""
    reauth_texts = []
    reauth_kws = ["reauthorization", "renewal criteria", "continuation criteria",
                  "continuation of therapy", "renewal approval", "re-authorization"]

    for i in relevant_page_indices:
        if i >= len(pages):
            continue
        lower = pages[i].lower()
        for kw in reauth_kws:
            if kw in lower:
                reauth_texts.append(f"[Page {i+1}]\n{pages[i].strip()}")
                break

    # Also check pages adjacent to relevant pages
    for i in range(len(pages)):
        if i in relevant_page_indices:
            continue
        lower = pages[i].lower()
        for kw in reauth_kws:
            if kw in lower:
                # Check if this page also mentions psoriasis or the target brand
                if any(pk in lower for pk in PRIMARY_KEYWORDS + ["plaque"]):
                    reauth_texts.append(f"[Page {i+1}]\n{pages[i].strip()}")
                    break

    return "\n\n".join(reauth_texts)


def extract_quantity_limit_text(pages: list[str], relevant_page_indices: set[int]) -> str:
    """
    Extract ONLY text explicitly labeled as quantity limits.
    Must NOT extract dosing/dosage information.
    """
    ql_texts = []
    for i in relevant_page_indices:
        if i >= len(pages):
            continue
        lower = pages[i].lower()
        if "quantity limit" in lower or "quantity level limit" in lower:
            ql_texts.append(f"[Page {i+1}]\n{pages[i].strip()}")

    return "\n\n".join(ql_texts)


def _clean_context_text(text: str) -> str:
    """Clean extracted PDF text to reduce token waste.

    Removes formatting artifacts without altering policy content:
    - Collapse runs of blank lines to a single blank line
    - Strip leading/trailing whitespace per line
    - Collapse multiple spaces/tabs to a single space
    - Remove repeated header/footer lines (date stamps, page numbers)
    - Remove standalone page numbers
    - Fix PDF extraction artifacts (spurious spaces in hyphenated words)
    """
    # Strip per-line padding
    lines = [l.strip() for l in text.split("\n")]
    text = "\n".join(lines)

    # Collapse multiple blank lines → single blank line
    text = re.sub(r"\n\s*\n(\s*\n)+", "\n\n", text)

    # Collapse inline multiple spaces/tabs → single space
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Remove repeated header/footer boilerplate
    text = re.sub(
        r"^(Last Update:.*|Effective:.*Page.*|Page\s*\d+\s*of\s*\d+.*)$",
        "", text, flags=re.MULTILINE,
    )

    # Remove standalone page numbers
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)

    # Fix common PDF artifact: spurious space before/after hyphens in compound words
    # e.g. "Authori zation" stays (can't fix mid-word breaks safely),
    # but "infliximab - axxq" → "infliximab-axxq"
    text = re.sub(r"(\w) - (\w)", r"\1-\2", text)
    text = re.sub(r"(\w)- (\w)", r"\1-\2", text)
    text = re.sub(r"(\w) -(\w)", r"\1-\2", text)

    # Clean up after removals
    text = re.sub(r"\n\s*\n(\s*\n)+", "\n\n", text)

    return text.strip()


# ── Paragraph / sentence splitting helpers ─────────────────────────────

def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs.

    PDF-extracted text rarely has double-newlines, so we also split on
    structural cues: numbered list items, section headers (lines ending
    with a colon followed by a newline), and blank-line-like gaps.
    """
    # First try double-newline split
    paras = re.split(r'\n\s*\n', text)
    if len(paras) > 1:
        return [p.strip() for p in paras if p.strip()]

    # PDF text: split on structural cues
    # - Numbered items: "1.", "2.", "a.", "A."
    # - Section headers: line ending with ":" followed by content
    # - Bullet points
    parts = re.split(
        r'(?:\n(?=\d+[.)]\s)|\n(?=[a-zA-Z][.)]\s)|\n(?=•\s)|\n(?=[-–]\s))',
        text
    )
    result = [p.strip() for p in parts if p.strip()]

    # If still just one big block, split on sentence boundaries to get
    # chunks of ~500 chars (roughly 2-3 sentences)
    if len(result) == 1 and len(result[0]) > 1000:
        sentences = _split_sentences(result[0])
        chunks = []
        current = []
        current_len = 0
        for sent in sentences:
            current.append(sent)
            current_len += len(sent)
            if current_len >= 500:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
        if current:
            chunks.append(" ".join(current))
        return chunks

    return result


def _is_literature_review(text: str, target_brand: str = "") -> bool:
    """Detect if text is a clinical literature review (not PA policy content).

    Literature reviews cite studies, mention trial designs, patient counts,
    and statistical outcomes. PA policy text mentions authorization criteria,
    step therapy, formulary tiers, and coverage rules.

    If target_brand is provided and the text mentions it (or its generic),
    the text is treated as policy content — payers sometimes cite a study
    inline to justify criteria.
    """
    lower = text.lower()

    # Brand-guard: if the target brand or its generic appears, this is
    # policy content that happens to reference a study, not a lit review.
    if target_brand:
        brand_lower = target_brand.lower()
        generic = BRAND_TO_GENERIC.get(target_brand, "").lower()
        if brand_lower in lower or (generic and generic in lower):
            return False

    # Literature signals
    lit_patterns = [
        r'et al', r'\(\d{4}\)', r'\bstudy\b', r'\btrial\b',
        r'\brandomized\b', r'\bplacebo\b', r'\befficacy\b',
        r'patients.*n\s*=', r'meta-analysis', r'systematic review',
        r'case.?series', r'\bcohort\b', r'\bresponders?\b',
        r'double.?blind', r'phase\s+[iI]{1,3}',
    ]
    lit_count = sum(len(re.findall(p, lower)) for p in lit_patterns)

    # Policy signals
    pol_patterns = [
        r'authorization', r'\bcriteria\b', r'\bapproval\b',
        r'step therapy', r'prior auth', r'precertification',
        r'medically necessary', r'\bcoverage\b', r'\bformulary\b',
        r'\bpreferred\b', r'non-preferred', r'reauthorization',
        r'\brenewal\b', r'quantity limit', r'\btier\b',
    ]
    pol_count = sum(len(re.findall(p, lower)) for p in pol_patterns)

    # Literature if: many lit signals AND few/no policy signals
    if lit_count > 10 and pol_count <= 2:
        return True
    if lit_count > 5 and pol_count == 0:
        return True
    # High ratio with meaningful lit count
    if lit_count >= 8 and lit_count / max(pol_count, 1) > 5:
        return True

    return False


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences. Handles abbreviations and numbered lists."""
    # Normalize newlines within paragraphs (PDF line breaks aren't sentence ends)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Split on sentence-ending punctuation followed by space + uppercase/number
    parts = re.split(r'(?<=[.;])\s+(?=[A-Z0-9(])', text)
    # Also split on numbered list items
    expanded = []
    for part in parts:
        sub = re.split(r'(?:^|\n)\s*\d+[.)]\s+', part)
        expanded.extend(sub)
    return [s.strip() for s in expanded if len(s.strip()) >= 20]


def _is_pso_relevant(text_lower: str) -> bool:
    """Check if text is about PsO (not PsA/UC/CD)."""
    has_pso = has_pso_signal(text_lower)
    has_non_pso = any(kw in text_lower for kw in NON_PSO_INDICATION_KEYWORDS)
    # If it mentions a non-PsO indication but NOT PsO, exclude it
    if has_non_pso and not has_pso:
        return False
    return True


def _detect_universal_boundary(pages: list[str]) -> int:
    """
    Detect where the universal/preamble section ends and drug-specific
    sections begin. Returns the page index (0-based) of the first
    drug-specific page.

    Capped at 10 pages to prevent runaway inclusion on large formularies.
    """
    max_preamble = min(10, len(pages))

    drug_section_patterns = [
        r'^products?\s+affected',
        r'^drug\s+name\s*:',
        r'^\s*(?:brand|generic)\s+name\s*:',
        r'(?:plaque\s+psoriasis|psoriatic\s+arthritis|crohn|ulcerative\s+colitis)',
    ]
    for i, page in enumerate(pages[:max_preamble]):
        lower = page.lower().strip()
        if len(lower) < 100:
            continue
        for pattern in drug_section_patterns:
            if re.search(pattern, lower[:500], re.MULTILINE):
                return max(i, 1)
    # No clear boundary found — assume first 3 pages or all if short
    return min(3, len(pages))


def _extract_tier1_universal(pages: list[str], doc_type: str,
                             brand: str) -> str:
    """
    Tier 1: Extract universal criteria text using structural detection.
    Not filtered by disease/brand keywords — filtered by section structure.
    """
    if doc_type in ("single-drug", "flat-catalog", "decision-tree"):
        return ""

    boundary = _detect_universal_boundary(pages)
    universal_parts = []

    # Preamble pages (before drug-specific sections)
    for i in range(min(boundary, len(pages))):
        text = pages[i].strip()
        if text:
            universal_parts.append(f"[Universal | Page {i+1}]\n{text}")

    # Also scan all pages for explicit universal section headers
    for i in range(boundary, len(pages)):
        lower = pages[i].lower()
        is_universal = False
        universal_headers = [
            r"general\s+authorization\s+guidelines",
            r"general\s+criteria",
            r"criteria\s+for\s+all",
            r"applies?\s+to\s+all",
            r"for\s+all\s+(medications|indications|requests|diagnoses)",
            r"documentation\s+for\s+all\s+indications",
        ]
        for pattern in universal_headers:
            if re.search(pattern, lower):
                is_universal = True
                break
        # Non-preferred tier requirements
        if not is_universal:
            tier_patterns = [
                r"non-preferred.*require.*trial.*(?:preferred|formulary)",
                r"non-preferred.*(?:agents?|products?).*require",
            ]
            for pattern in tier_patterns:
                if re.search(pattern, lower):
                    is_universal = True
                    break
        if is_universal:
            universal_parts.append(f"[Universal | Page {i+1}]\n{pages[i].strip()}")

    return "\n\n".join(universal_parts)


def _extract_tier2_pso_section(pages: list[str], brand: str,
                               doc_type: str) -> tuple[str, list[int]]:
    """
    Tier 2: Extract PsO + brand content at paragraph granularity.

    For single-drug docs: include all pages (the whole doc is relevant).
    For multi-drug docs: find paragraphs that are about PsO criteria for
    the target brand, filtering out non-PsO indications (UC, CD, PsA
    literature reviews, etc.).

    Returns (text, list of page indices that contributed content).
    """
    brand_lower = brand.lower()
    generic = BRAND_TO_GENERIC.get(brand, "").lower()

    # For single-drug docs, include everything
    if doc_type == "single-drug":
        parts = []
        indices = []
        for i, page in enumerate(pages):
            text = page.strip()
            if text:
                parts.append(f"[PsO Section | Page {i+1}]\n{text}")
                indices.append(i)
        return "\n\n".join(parts), indices

    # Multi-drug: paragraph-level filtering with literature exclusion
    all_param_kws = []
    for kws in PARAMETER_PATTERNS.values():
        all_param_kws.extend(kws)

    kept: list[tuple[int, str]] = []  # (page_index, paragraph)
    page_indices_used: set[int] = set()

    for i, page in enumerate(pages):
        # Skip entire pages that are clearly literature reviews
        if _is_literature_review(page, target_brand=brand):
            continue

        paras = _split_paragraphs(page)
        for para in paras:
            if len(para) < 30:
                continue
            lower = para.lower()

            # Skip literature-review paragraphs
            if len(para) > 300 and _is_literature_review(para, target_brand=brand):
                continue

            has_pso = has_pso_signal(lower)
            has_brand = brand_lower in lower or (generic and generic in lower)
            has_param = any(kw in lower for kw in all_param_kws)

            # Inclusion: must relate to PsO criteria for this brand
            include = False
            if has_pso and has_brand:
                include = True
            elif has_pso and has_param:
                include = True
            elif has_brand and has_param:
                include = _is_pso_relevant(lower)

            if include and not _is_pso_relevant(lower):
                include = False

            if include:
                kept.append((i, para))
                page_indices_used.add(i)

    if not kept:
        # Fallback: any paragraph mentioning PsO (skip literature)
        for i, page in enumerate(pages):
            if _is_literature_review(page, target_brand=brand):
                continue
            for para in _split_paragraphs(page):
                lower = para.lower()
                if has_pso_signal(lower) and len(para) >= 30:
                    kept.append((i, para))
                    page_indices_used.add(i)

    section_parts = []
    for page_idx, para in kept:
        section_parts.append(f"[PsO Section | Page {page_idx + 1}]\n{para}")

    return "\n\n".join(section_parts), sorted(page_indices_used)


def _extract_tier3_snippets(pages: list[str], brand: str,
                            tier2_page_indices: list[int]) -> str:
    """
    Tier 3: Parameter-specific micro-extractions.
    For each parameter, find sentences/paragraphs containing its keywords
    that also mention the brand or PsO. Extract 2-3 sentences of context
    around each match. Only pulls from pages NOT already in Tier 2.
    """
    brand_lower = brand.lower()
    generic = BRAND_TO_GENERIC.get(brand, "").lower()
    tier2_set = set(tier2_page_indices)

    snippets_by_param: dict[str, list[str]] = {k: [] for k in PARAMETER_PATTERNS}
    seen_snippets: set[str] = set()  # deduplicate

    for i, page in enumerate(pages):
        # Skip pages already fully included in Tier 2
        if i in tier2_set:
            continue

        # Skip literature review pages
        if _is_literature_review(page, target_brand=brand):
            continue

        lower = page.lower()
        has_brand = brand_lower in lower or (generic and generic in lower)
        has_pso = has_pso_signal(lower)

        # Only extract from pages that mention brand or PsO
        if not has_brand and not has_pso:
            continue

        sentences = _split_sentences(page)

        for param, keywords in PARAMETER_PATTERNS.items():
            for j, sent in enumerate(sentences):
                sent_lower = sent.lower()
                if any(kw in sent_lower for kw in keywords):
                    # Skip if this sentence is about a non-PsO indication
                    if not _is_pso_relevant(sent_lower):
                        continue

                    # Extract the matching sentence + 1 sentence before/after for context
                    context_start = max(0, j - 1)
                    context_end = min(len(sentences), j + 2)
                    snippet = " ".join(sentences[context_start:context_end])

                    # Deduplicate
                    snippet_key = snippet[:100]
                    if snippet_key in seen_snippets:
                        continue
                    seen_snippets.add(snippet_key)

                    snippets_by_param[param].append(snippet)

    # Assemble Tier 3 text
    parts = []
    for param, snips in snippets_by_param.items():
        if snips:
            param_label = param.upper().replace("_", " ")
            combined = "\n".join(f"  - {s}" for s in snips[:5])  # cap at 5 per param
            parts.append(f"[{param_label}]\n{combined}")

    return "\n\n".join(parts)


def _build_page_level(filename: str, brand: str, pages: list[str],
                      doc_type: str, preferred_status: str) -> ContextPackage:
    """Original page-level filtering (filter_level='page')."""
    n_pages = len(pages)

    # Score every page
    page_infos = []
    for i, text in enumerate(pages):
        score, categories = score_page_relevance(text, brand)
        tag = "irrelevant"
        if score >= 3.0:
            tag = "relevant"
        elif "universal" in categories:
            tag = "universal"
        page_infos.append(PageInfo(
            page_num=i, text=text, tag=tag, relevance_score=score
        ))

    # Tag context pages (adjacent to relevant pages)
    relevant_indices = {pi.page_num for pi in page_infos if pi.tag == "relevant"}
    for pi in page_infos:
        if pi.tag == "irrelevant":
            if (pi.page_num - 1) in relevant_indices or (pi.page_num + 1) in relevant_indices:
                pi.tag = "context"

    # Tag preamble pages for multi-drug docs
    if doc_type == "multi-drug":
        for pi in page_infos[:5]:
            if pi.tag == "irrelevant":
                pi.tag = "preamble"

    # For large PDFs, stricter relevance
    if n_pages > 50:
        for pi in page_infos:
            if pi.tag == "relevant":
                _, categories = score_page_relevance(pi.text, brand)
                has_primary = "disease" in categories or "target_brand" in categories
                if not has_primary:
                    pi.tag = "irrelevant"
                    pi.relevance_score = 0
        relevant_indices = {pi.page_num for pi in page_infos if pi.tag == "relevant"}
        for pi in page_infos:
            if pi.tag == "context":
                if not ((pi.page_num - 1) in relevant_indices or (pi.page_num + 1) in relevant_indices):
                    pi.tag = "irrelevant"

    # Determine which pages to include
    if doc_type == "single-drug" or n_pages <= 15:
        included_pages = page_infos
    else:
        included_pages = [
            pi for pi in page_infos
            if pi.tag in ("relevant", "context", "preamble", "universal")
        ]
        if len([p for p in included_pages if p.tag == "relevant"]) < 2:
            included_pages = [
                pi for pi in page_infos
                if pi.relevance_score > 0 or pi.tag in ("preamble", "universal")
            ]

    included_pages.sort(key=lambda p: p.page_num)
    all_included_indices = {pi.page_num for pi in included_pages}

    universal_text = extract_universal_criteria(pages, doc_type, brand)
    reauth_text = extract_reauthorization_text(pages, all_included_indices)
    ql_text = extract_quantity_limit_text(pages, all_included_indices)

    pso_texts = []
    for pi in included_pages:
        if pi.tag in ("relevant", "context"):
            pso_texts.append(f"[Page {pi.page_num + 1}]\n{pi.text.strip()}")
    psoriasis_text = "\n\n".join(pso_texts)

    full_texts = []
    for pi in included_pages:
        full_texts.append(f"[Page {pi.page_num + 1} | Tag: {pi.tag}]\n{pi.text.strip()}")
    full_relevant_text = "\n\n---\n\n".join(full_texts)

    return ContextPackage(
        filename=filename,
        brand=brand,
        universal_criteria_text=universal_text,
        psoriasis_section_text=psoriasis_text,
        brand_specific_text="",
        reauthorization_text=reauth_text,
        quantity_limit_text=ql_text,
        preferred_status=preferred_status,
        document_type=doc_type,
        total_pages=n_pages,
        relevant_pages_used=len(included_pages),
        full_relevant_text=_clean_context_text(full_relevant_text),
    )


def _build_paragraph_level(filename: str, brand: str, pages: list[str],
                           doc_type: str, preferred_status: str) -> ContextPackage:
    """Paragraph-level filtering (filter_level='paragraph').

    Splits each page into paragraphs and keeps only those matching
    PsO/brand/parameter keywords. Applies PsO vs PsA negative filter.
    """
    n_pages = len(pages)
    brand_lower = brand.lower()
    generic = BRAND_TO_GENERIC.get(brand, "").lower()

    # Tier 1: universal criteria (structural, not keyword-filtered)
    universal_text = _extract_tier1_universal(pages, doc_type, brand)

    # Tier 2+3 combined at paragraph level
    kept_paras: list[tuple[int, str]] = []  # (page_num, paragraph_text)
    all_param_kws = []
    for kws in PARAMETER_PATTERNS.values():
        all_param_kws.extend(kws)

    for i, page in enumerate(pages):
        # Skip literature review pages
        if _is_literature_review(page, target_brand=brand):
            continue

        paras = _split_paragraphs(page)
        for para in paras:
            if len(para) < 30:
                continue
            lower = para.lower()

            # Skip literature-review paragraphs
            if len(para) > 300 and _is_literature_review(para, target_brand=brand):
                continue

            has_pso = has_pso_signal(lower)
            has_brand = brand_lower in lower or (generic and generic in lower)
            has_param = any(kw in lower for kw in all_param_kws)

            # Keep if: (PsO AND brand) OR (PsO AND param keyword) OR (brand AND param keyword)
            keep = False
            if has_pso and has_brand:
                keep = True
            elif has_pso and has_param:
                keep = True
            elif has_brand and has_param:
                keep = True

            if keep and _is_pso_relevant(lower):
                kept_paras.append((i, para))

    # Assemble
    section_parts = []
    page_indices_used = set()
    for page_num, para in kept_paras:
        page_indices_used.add(page_num)
        section_parts.append(f"[Page {page_num + 1}]\n{para}")

    psoriasis_text = "\n\n".join(section_parts)

    # Build full text: universal + filtered paragraphs
    full_parts = []
    if universal_text:
        full_parts.append(universal_text)
    if section_parts:
        full_parts.append("\n\n".join(section_parts))

    full_relevant_text = "\n\n---\n\n".join(full_parts)

    return ContextPackage(
        filename=filename,
        brand=brand,
        universal_criteria_text=universal_text,
        psoriasis_section_text=psoriasis_text,
        brand_specific_text="",
        reauthorization_text="",
        quantity_limit_text="",
        preferred_status=preferred_status,
        document_type=doc_type,
        total_pages=n_pages,
        relevant_pages_used=len(page_indices_used),
        full_relevant_text=_clean_context_text(full_relevant_text),
    )


def _build_sentence_level(filename: str, brand: str, pages: list[str],
                          doc_type: str, preferred_status: str) -> ContextPackage:
    """Sentence-level 3-tier extraction (filter_level='sentence').

    Tier 1: Universal criteria via structural detection (not keyword-filtered).
    Tier 2: Contiguous PsO+brand section (paragraph-level, with stop-at-new-section).
    Tier 3: Parameter-specific sentence snippets from remaining pages.
    """
    n_pages = len(pages)

    # Tier 1
    universal_text = _extract_tier1_universal(pages, doc_type, brand)

    # Tier 2
    tier2_text, tier2_pages = _extract_tier2_pso_section(pages, brand, doc_type)

    # For single-drug or small docs, Tier 2 already has everything — skip Tier 3
    if doc_type == "single-drug" or n_pages <= 10:
        full_parts = []
        if universal_text:
            full_parts.append(universal_text)
        if tier2_text:
            full_parts.append(tier2_text)
        full_relevant_text = "\n\n---\n\n".join(full_parts)

        return ContextPackage(
            filename=filename,
            brand=brand,
            universal_criteria_text=universal_text,
            psoriasis_section_text=tier2_text,
            brand_specific_text="",
            reauthorization_text="",
            quantity_limit_text="",
            preferred_status=preferred_status,
            document_type=doc_type,
            total_pages=n_pages,
            relevant_pages_used=len(tier2_pages) if tier2_pages else n_pages,
            full_relevant_text=_clean_context_text(full_relevant_text),
        )

    # Tier 3: parameter-specific snippets from pages outside Tier 2
    tier3_text = _extract_tier3_snippets(pages, brand, tier2_pages)

    # Assemble final context
    full_parts = []
    if universal_text:
        full_parts.append(universal_text)
    if tier2_text:
        full_parts.append(tier2_text)
    if tier3_text:
        full_parts.append(f"[PARAMETER-SPECIFIC SNIPPETS]\n{tier3_text}")

    full_relevant_text = "\n\n---\n\n".join(full_parts)

    # Count pages used across all tiers
    pages_used = set(tier2_pages) if tier2_pages else set()
    # Add pages from universal text
    for match in re.finditer(r'\[Universal \| Page (\d+)\]', universal_text):
        pages_used.add(int(match.group(1)) - 1)
    # Add pages from tier3 (approximate from snippet content)
    if tier3_text:
        pages_used.add(-1)  # sentinel to indicate tier3 contributed

    return ContextPackage(
        filename=filename,
        brand=brand,
        universal_criteria_text=universal_text,
        psoriasis_section_text=tier2_text,
        brand_specific_text="",
        reauthorization_text="",
        quantity_limit_text="",
        preferred_status=preferred_status,
        document_type=doc_type,
        total_pages=n_pages,
        relevant_pages_used=len(pages_used - {-1}),
        full_relevant_text=_clean_context_text(full_relevant_text),
    )


def build_context_package(
    filename: str,
    brand: str,
    pages: list[str],
    filter_level: str | None = None,
) -> ContextPackage:
    """
    Build a structured context package for one (PDF, Brand) pair.
    This is the main entry point for Stage 1+2.

    Args:
        filter_level: "page", "paragraph", or "sentence". Defaults to config.FILTER_LEVEL.
    """
    level = (filter_level or FILTER_LEVEL).lower()
    doc_type = classify_document(pages)
    preferred_status = detect_preferred_status(pages, brand)

    if level == "page":
        return _build_page_level(filename, brand, pages, doc_type, preferred_status)
    elif level == "paragraph":
        return _build_paragraph_level(filename, brand, pages, doc_type, preferred_status)
    elif level == "sentence":
        return _build_sentence_level(filename, brand, pages, doc_type, preferred_status)
    else:
        raise ValueError(f"Unknown filter_level: {level!r}. Use 'page', 'paragraph', or 'sentence'.")


def process_all_pdfs(submission_rows: list[tuple[str, str]],
                     filter_level: str | None = None) -> list[ContextPackage]:
    """
    Process all PDFs and build context packages for each (filename, brand) pair.

    Args:
        submission_rows: List of (filename, brand) tuples from the Submissions tab.
        filter_level: Override for context filter level ("page"/"paragraph"/"sentence").

    Returns:
        List of ContextPackage objects ready for LLM extraction.
    """
    level = filter_level or FILTER_LEVEL
    pdf_dir = get_pdf_dir()
    packages = []

    # Group by filename to avoid re-reading the same PDF
    from collections import defaultdict
    file_brands = defaultdict(list)
    for filename, brand in submission_rows:
        file_brands[filename].append(brand)

    print(f"  Filter level: {level}")

    for filename, brands in file_brands.items():
        pdf_path = os.path.join(pdf_dir, filename)
        if not os.path.isfile(pdf_path):
            print(f"WARNING: PDF not found: {pdf_path}")
            continue

        pages = extract_pdf_text(pdf_path)
        print(f"  {filename}: {len(pages)} pages, brands={brands}")

        for brand in brands:
            pkg = build_context_package(filename, brand, pages, filter_level=level)
            packages.append(pkg)
            print(f"    -> {brand}: {pkg.document_type}, {pkg.relevant_pages_used}/{pkg.total_pages} pages, "
                  f"preferred={pkg.preferred_status}, text={len(pkg.full_relevant_text)} chars")

    return packages

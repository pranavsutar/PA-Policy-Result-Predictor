#!/usr/bin/env python3
"""
debug.py — Single-PDF deep inspection tool.

Runs the full pipeline on one (file, brand) pair and prints every
intermediate artifact with clear labels. Saves all artifacts to
debug/{filename}_{brand}/ as separate files.

Usage:
  # Full debug (with Groq API call)
  python debug.py --file 330109-4880941.pdf --brand TREMFYA --api-key KEY

  # Stages 1+2 only (no API call)
  python debug.py --file 330109-4880941.pdf --brand TREMFYA

  # Dump prompt to file for manual testing in AI Studio
  python debug.py --file 330109-4880941.pdf --brand TREMFYA --prompt-export

  # Score breakdown only (requires cached extraction)
  python debug.py --file 330109-4880941.pdf --brand TREMFYA --score-only
"""
import argparse
import json
import os
import sys
import re

from config import get_pdf_dir, get_excel_path, CACHE_DIR, GROQ_API_KEYS, LITELLM_MODELS
from pdf_extractor import (
    extract_pdf_text, score_page_relevance, classify_document,
    detect_preferred_status, build_context_package, extract_universal_criteria,
)
from prompts import (
    SYSTEM_PROMPT, build_extraction_prompt, select_few_shot_examples,
    format_few_shot_examples,
)
from scorer import (
    compute_access_score, score_steps_brands, score_steps_generic,
    score_phototherapy, score_tb_test, score_age, score_init_auth_duration,
    score_reauth_required, score_quantity_limits, score_specialist,
    score_reauth_requirements,
)
from validator import normalize_output, enforce_business_rules


def ensure_debug_dir(filename: str, brand: str) -> str:
    """Create and return the debug output directory."""
    safe_name = re.sub(r'[^\w\-.]', '_', f"{filename}_{brand}")
    debug_dir = os.path.join(os.path.dirname(CACHE_DIR), "debug", safe_name)
    os.makedirs(debug_dir, exist_ok=True)
    return debug_dir


def save_artifact(debug_dir: str, name: str, content: str):
    """Save a debug artifact to file."""
    path = os.path.join(debug_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def section(title: str):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def debug_stage1_page_scoring(pages: list[str], brand: str, debug_dir: str):
    """Print and save page-level relevance scores."""
    section("STAGE 1: Page-Level Relevance Scoring")

    lines = []
    for i, text in enumerate(pages):
        score, categories = score_page_relevance(text, brand)
        char_count = len(text)
        preview = text[:100].replace('\n', ' ').strip()
        line = (f"  Page {i+1:3d}: score={score:5.1f}  cats={categories!s:50s}  "
                f"chars={char_count:5d}  preview={preview[:60]}...")
        print(line)
        lines.append(line)

    save_artifact(debug_dir, "01_page_scores.txt", "\n".join(lines))
    print(f"\n  Saved to: {debug_dir}/01_page_scores.txt")


def debug_stage2_classification(pages: list[str], brand: str, debug_dir: str):
    """Print document classification details."""
    section("STAGE 2: Document Classification")

    doc_type = classify_document(pages)
    preferred = detect_preferred_status(pages, brand)
    universal_text = extract_universal_criteria(pages, doc_type, brand)

    print(f"  Document type: {doc_type}")
    print(f"  Preferred status: {preferred}")
    print(f"  Total pages: {len(pages)}")
    print(f"  Universal criteria text: {len(universal_text)} chars")

    if universal_text:
        print(f"\n  --- Universal Criteria (first 1000 chars) ---")
        print(f"  {universal_text[:1000]}")
        save_artifact(debug_dir, "02_universal_criteria.txt", universal_text)
    else:
        print(f"  (No universal criteria detected)")

    info = {
        "document_type": doc_type,
        "preferred_status": preferred,
        "total_pages": len(pages),
        "universal_criteria_chars": len(universal_text),
    }
    save_artifact(debug_dir, "02_classification.json", json.dumps(info, indent=2))


def debug_context_package(pkg, debug_dir: str):
    """Print context package summary."""
    section("STAGE 2: Context Package")

    print(f"  Filename: {pkg.filename}")
    print(f"  Brand: {pkg.brand}")
    print(f"  Document type: {pkg.document_type}")
    print(f"  Preferred status: {pkg.preferred_status}")
    print(f"  Total pages: {pkg.total_pages}")
    print(f"  Relevant pages used: {pkg.relevant_pages_used}")
    print(f"  Full text: {len(pkg.full_relevant_text)} chars (~{len(pkg.full_relevant_text)//4} tokens)")
    print(f"  Psoriasis section: {len(pkg.psoriasis_section_text)} chars")
    print(f"  Reauthorization text: {len(pkg.reauthorization_text)} chars")
    print(f"  Quantity limit text: {len(pkg.quantity_limit_text)} chars")

    # Extract page numbers from the full text
    page_nums = re.findall(r'\[Page (\d+)', pkg.full_relevant_text)
    print(f"  Pages included: {page_nums}")

    save_artifact(debug_dir, "03_full_relevant_text.txt", pkg.full_relevant_text)
    save_artifact(debug_dir, "03_psoriasis_section.txt", pkg.psoriasis_section_text)
    save_artifact(debug_dir, "03_reauth_text.txt", pkg.reauthorization_text)
    save_artifact(debug_dir, "03_quantity_limit_text.txt", pkg.quantity_limit_text)

    summary = {
        "filename": pkg.filename,
        "brand": pkg.brand,
        "document_type": pkg.document_type,
        "preferred_status": pkg.preferred_status,
        "total_pages": pkg.total_pages,
        "relevant_pages_used": pkg.relevant_pages_used,
        "full_text_chars": len(pkg.full_relevant_text),
        "pages_included": page_nums,
    }
    save_artifact(debug_dir, "03_context_summary.json", json.dumps(summary, indent=2))
    print(f"\n  Artifacts saved to: {debug_dir}/03_*")


def debug_prompt(pkg, debug_dir: str):
    """Build and print the exact prompt."""
    section("PROMPT")

    examples = select_few_shot_examples(pkg.brand, pkg.document_type)
    few_shot_text = format_few_shot_examples(examples)
    extraction_prompt = build_extraction_prompt(pkg)
    full_prompt = f"{few_shot_text}\n\n---\n\n{extraction_prompt}"

    print(f"  System prompt: {len(SYSTEM_PROMPT)} chars")
    print(f"  Few-shot examples: {len(examples)} selected")
    print(f"  User prompt: {len(full_prompt)} chars (~{len(full_prompt)//4} tokens)")
    print(f"  Total: ~{(len(SYSTEM_PROMPT) + len(full_prompt))//4} tokens")

    save_artifact(debug_dir, "04_system_prompt.txt", SYSTEM_PROMPT)
    save_artifact(debug_dir, "04_user_prompt.txt", full_prompt)
    save_artifact(debug_dir, "04_full_prompt.txt",
                  f"=== SYSTEM ===\n{SYSTEM_PROMPT}\n\n=== USER ===\n{full_prompt}")

    print(f"\n  Prompt saved to: {debug_dir}/04_full_prompt.txt")
    print(f"  (Copy-paste into Google AI Studio for manual testing)")

    return full_prompt


def debug_extraction(pkg, api_key: str, debug_dir: str,
                     litellm_models: list | None = None) -> dict | None:
    """Run extraction and print raw + parsed results."""
    section("STAGE 3 PASS 1: Extraction")

    if not api_key and not litellm_models:
        print("  Skipped (no API key). Use --api-key or --litellm to run extraction.")
        # Try loading from cache
        from extractor import StandardExtractor, _cache_key
        ext = StandardExtractor(api_key="", use_cache=True)
        key = _cache_key(pkg.filename, pkg.brand, "extract")
        cached = ext._load_cache(key)
        if cached:
            print(f"  Loaded from cache: {key}")
            print(f"  {json.dumps(cached, indent=2)}")
            save_artifact(debug_dir, "05_extraction.json", json.dumps(cached, indent=2))
            return cached
        print("  No cache found either.")
        return None

    from extractor import StandardExtractor
    ext = StandardExtractor(api_key=api_key or "", use_cache=False,
                          litellm_models=litellm_models)
    result = ext.extract_single(pkg)

    print(f"  API calls used: {ext.api_calls_made}")
    print(f"  Result:")
    print(f"  {json.dumps(result, indent=2)}")

    save_artifact(debug_dir, "05_extraction.json", json.dumps(result, indent=2))
    return result


def debug_validation(pkg, extracted: dict, api_key: str, debug_dir: str,
                     litellm_models: list | None = None) -> dict:
    """Run validation and print corrections."""
    section("STAGE 3 PASS 2: Validation")

    if not api_key and not litellm_models:
        print("  Skipped (no API key).")
        # Try loading from cache
        from extractor import StandardExtractor, _cache_key
        ext = StandardExtractor(api_key="", use_cache=True)
        key = _cache_key(pkg.filename, pkg.brand, "validate")
        cached = ext._load_cache(key)
        if cached:
            print(f"  Loaded from cache: {key}")
            corrections = cached.get("corrections", "None")
            print(f"  Corrections: {corrections}")
            save_artifact(debug_dir, "06_validated.json", json.dumps(cached, indent=2))
            return cached
        print("  No cache found. Using extraction result as-is.")
        return extracted

    from extractor import StandardExtractor
    ext = StandardExtractor(api_key=api_key or "", use_cache=False,
                          litellm_models=litellm_models)
    validated = ext.validate_single(pkg, extracted)

    corrections = validated.get("corrections", "None")
    print(f"  API calls used: {ext.api_calls_made}")
    print(f"  Corrections: {corrections}")
    print(f"  Validated result:")
    print(f"  {json.dumps(validated, indent=2)}")

    save_artifact(debug_dir, "06_validated.json", json.dumps(validated, indent=2))

    # Show diff
    diffs = []
    for key in extracted:
        if key in validated and str(extracted[key]) != str(validated[key]):
            diffs.append((key, extracted[key], validated[key]))
    if diffs:
        print(f"\n  Changes from validation:")
        for key, old, new in diffs:
            print(f"    {key}: {str(old)[:60]} → {str(new)[:60]}")
    else:
        print(f"\n  No changes from validation.")

    return validated


def debug_business_rules(data: dict, filename: str, brand: str, debug_dir: str) -> dict:
    """Run normalization and business rule enforcement."""
    section("POST-PROCESSING: Business Rules")

    row = normalize_output(data, filename, brand)
    corrected, violations = enforce_business_rules(row)

    print(f"  Normalized output:")
    for col, val in corrected.items():
        print(f"    {col}: {val}")

    if violations:
        print(f"\n  Business rule violations ({len(violations)}):")
        for v in violations:
            print(f"    ❌ {v}")
    else:
        print(f"\n  ✅ No business rule violations.")

    save_artifact(debug_dir, "07_normalized.json", json.dumps(corrected, indent=2))
    if violations:
        save_artifact(debug_dir, "07_violations.txt", "\n".join(violations))

    return corrected


def debug_score_breakdown(data: dict, brand: str, debug_dir: str):
    """Print detailed Access Score breakdown."""
    section("STAGE 4: Access Score Breakdown")

    components = [
        ("Steps through Brands", "steps_through_brands",
         score_steps_brands(data.get("steps_through_brands")), 20),
        ("Steps through Generic", "steps_through_generic",
         score_steps_generic(data.get("steps_through_generic")), 15),
        ("Phototherapy", "step_through_phototherapy",
         score_phototherapy(data.get("step_through_phototherapy")), 5),
        ("TB Test", "tb_test_required",
         score_tb_test(data.get("tb_test_required")), 5),
        ("Age", "age",
         score_age(data.get("age"), brand), 10),
        ("Init Auth Duration", "initial_auth_duration",
         score_init_auth_duration(data.get("initial_auth_duration")), 15),
        ("Reauth Required", "reauth_required/reauth_duration",
         score_reauth_required(data.get("reauth_required"), data.get("reauth_duration")), 10),
        ("Quantity Limits", "quantity_limits",
         score_quantity_limits(data.get("quantity_limits")), 5),
        ("Specialist", "specialist_types",
         score_specialist(data.get("specialist_types")), 5),
        ("Reauth Requirements", "reauth_requirements",
         score_reauth_requirements(data.get("reauth_requirements")), 10),
    ]

    total = 0
    lines = []
    print(f"  {'Component':<25s} {'Value':<30s} {'Points':>6s} {'Max':>4s}")
    print(f"  {'-'*25} {'-'*30} {'-'*6} {'-'*4}")

    for label, key, points, max_pts in components:
        if "/" in key:
            keys = key.split("/")
            value = " / ".join(str(data.get(k, "?")) for k in keys)
        else:
            value = str(data.get(key, "?"))
        value_display = value[:30]
        line = f"  {label:<25s} {value_display:<30s} {points:>6d} {max_pts:>4d}"
        print(line)
        lines.append(line)
        total += points

    print(f"  {'-'*25} {'-'*30} {'-'*6} {'-'*4}")
    total_line = f"  {'TOTAL':<25s} {'':30s} {total:>6d}  100"
    print(total_line)
    lines.append(total_line)

    final_score = compute_access_score(data, brand)
    print(f"\n  Final Access Score: {final_score}")

    save_artifact(debug_dir, "08_score_breakdown.txt", "\n".join(lines) + f"\n\nFinal: {final_score}")


def main():
    parser = argparse.ArgumentParser(description="Single-PDF deep inspection tool")
    parser.add_argument("--file", required=True, help="PDF filename")
    parser.add_argument("--brand", required=True, help="Brand name (e.g., TREMFYA)")
    parser.add_argument("--api-key", default=None,
                        help="Groq API key(s) (optional — stages 1+2 run without it)")
    parser.add_argument("--litellm", default=None,
                        help="LiteLLM fallback model:key pairs, comma-separated "
                             "(e.g., 'gpt-4o-mini:sk-abc,claude-sonnet-4-20250514:sk-xyz')")
    parser.add_argument("--prompt-export", action="store_true",
                        help="Export prompt to file and exit")
    parser.add_argument("--score-only", action="store_true",
                        help="Show score breakdown only (requires cached extraction)")

    args = parser.parse_args()

    # Resolve API key: CLI > .env
    api_key = args.api_key
    if not api_key and GROQ_API_KEYS:
        api_key = ",".join(GROQ_API_KEYS)

    # Resolve LiteLLM models: CLI > .env
    litellm_models = None
    if args.litellm:
        litellm_models = []
        for entry in args.litellm.split(","):
            entry = entry.strip()
            if ":" in entry:
                m, k = entry.split(":", 1)
                if m.strip() and k.strip():
                    litellm_models.append((m.strip(), k.strip()))
    elif LITELLM_MODELS:
        litellm_models = list(LITELLM_MODELS)

    pdf_dir = get_pdf_dir()
    pdf_path = os.path.join(pdf_dir, args.file)
    if not os.path.isfile(pdf_path):
        print(f"ERROR: PDF not found: {pdf_path}")
        sys.exit(1)

    debug_dir = ensure_debug_dir(args.file, args.brand)
    print(f"Debug output directory: {debug_dir}")

    # Score-only mode
    if args.score_only:
        from extractor import StandardExtractor, _cache_key
        from pathlib import Path
        cache_dir = Path(CACHE_DIR)
        ext = StandardExtractor(api_key="", use_cache=True)
        for pass_name in ("validate", "extract"):
            key = _cache_key(args.file, args.brand, pass_name)
            cached = ext._load_cache(key)
            if cached:
                debug_score_breakdown(cached, args.brand, debug_dir)
                debug_business_rules(cached, args.file, args.brand, debug_dir)
                return
        print("ERROR: No cached extraction found. Run extraction first.")
        sys.exit(1)

    # Stage 1: Extract text
    section("STAGE 1: PDF Text Extraction")
    pages = extract_pdf_text(pdf_path)
    print(f"  Pages extracted: {len(pages)}")
    print(f"  Total chars: {sum(len(p) for p in pages):,}")

    # Page scoring
    debug_stage1_page_scoring(pages, args.brand, debug_dir)

    # Stage 2: Classification + context package
    debug_stage2_classification(pages, args.brand, debug_dir)
    pkg = build_context_package(args.file, args.brand, pages)
    debug_context_package(pkg, debug_dir)

    # Prompt
    full_prompt = debug_prompt(pkg, debug_dir)

    if args.prompt_export:
        print(f"\n  Prompt exported. Open {debug_dir}/04_full_prompt.txt")
        print(f"  Paste into Google AI Studio to test manually.")
        return

    # Stage 3: Extraction
    extracted = debug_extraction(pkg, api_key, debug_dir, litellm_models=litellm_models)
    if extracted is None:
        print("\n  Cannot continue without extraction. Provide --api-key, --litellm, "
              "or set GROQ_API_KEYS / LITELLM_MODELS in .env")
        return

    # Stage 3: Validation
    validated = debug_validation(pkg, extracted, api_key, debug_dir,
                                 litellm_models=litellm_models)

    # Business rules
    corrected = debug_business_rules(validated, args.file, args.brand, debug_dir)

    # Score breakdown
    debug_score_breakdown(validated, args.brand, debug_dir)

    section("COMPLETE")
    print(f"  All artifacts saved to: {debug_dir}/")
    print(f"  Files:")
    for f in sorted(os.listdir(debug_dir)):
        size = os.path.getsize(os.path.join(debug_dir, f))
        print(f"    {f} ({size:,} bytes)")


if __name__ == "__main__":
    main()

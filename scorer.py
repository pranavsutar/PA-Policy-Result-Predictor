"""
scorer.py — Stage 4: Deterministic Access Score computation (0-100).
Condition-based scoring, not character-length heuristics.
"""
import re

from config import FDA_LABELED_AGE

# ── Tunable offset ──────────────────────────────────────────────────────────
# Adjust this single value to shift the entire score distribution up or down.
#   +15 → scores land around 25/50/75
#   +0  → scores land around 0/25/50
#   +25 → scores land around 50/75/100
# Does not affect hard-floor cases (extreme step therapy → 0 or 25).
SCORE_OFFSET = 25


def _parse_age_numeric(age_str: str) -> int | None:
    """Extract numeric age from strings like '>=18', '>=6', '<18'."""
    if not age_str or age_str in ("No", "NA", "Unspecified", "FDA labelled age",
                                   "FDA approved age", "FDA labeled age"):
        return None
    match = re.search(r'(\d+)', str(age_str))
    if match:
        return int(match.group(1))
    return None


def _parse_numeric(value) -> int | None:
    """Parse a value that should be numeric. Returns None for non-numeric."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip()
    if s in ("NA", "N/A", "No", "Unspecified", ""):
        return None
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def _is_na_or_no(value) -> bool:
    """Check if a value represents NA/No/empty."""
    if value is None:
        return True
    s = str(value).strip()
    return s in ("NA", "N/A", "No", "", "0")


def score_steps_brands(value) -> int:
    """Score branded step count. Weight: 20."""
    if _is_na_or_no(value):
        return 20
    n = _parse_numeric(value)
    if n is None:
        return 20  # NA-like
    if n == 0:
        return 20
    if n == 1:
        return 12
    if n == 2:
        return 6
    return 0  # 3+


def score_steps_generic(value) -> int:
    """Score generic step count. Weight: 15."""
    if _is_na_or_no(value):
        return 15
    n = _parse_numeric(value)
    if n is None:
        return 15
    if n == 0:
        return 15
    if n == 1:
        return 10
    if n == 2:
        return 5
    return 0  # 3+


def score_phototherapy(value) -> int:
    """Score phototherapy requirement. Weight: 5."""
    s = str(value).strip() if value else ""
    if s in ("Yes",):
        return 0
    return 5  # No, N/A, or empty


def score_tb_test(value) -> int:
    """Score TB test requirement. Weight: 5."""
    s = str(value).strip() if value else ""
    if s in ("Yes", "Y"):
        return 0
    return 5


def score_age(value, brand: str) -> int:
    """
    Score age restriction. Weight: 10.
    - No restriction → 10
    - Matches FDA label → 8
    - More restrictive than FDA → 3
    """
    s = str(value).strip() if value else "No"

    if s in ("No", ""):
        return 10

    if s in ("FDA labelled age", "FDA approved age", "FDA labeled age"):
        return 8

    extracted_age = _parse_age_numeric(s)
    if extracted_age is None:
        return 8  # Can't parse, assume FDA-like

    # Compare against FDA labeled age
    fda_age_str = FDA_LABELED_AGE.get(brand, ">=18")
    fda_age = _parse_age_numeric(fda_age_str)
    if fda_age is None:
        fda_age = 18

    if extracted_age == fda_age:
        return 8
    elif extracted_age > fda_age:
        return 3  # More restrictive
    else:
        return 10  # Less restrictive (broader access)


def score_init_auth_duration(value) -> int:
    """Score initial authorization duration. Weight: 15."""
    s = str(value).strip() if value else "Unspecified"

    if s == "Unspecified":
        return 8

    n = _parse_numeric(value)
    if n is None:
        return 8  # Unspecified-like
    if n >= 12:
        return 15
    if n >= 6:
        return 10
    if n >= 3:
        return 5
    return 3  # < 3 months


def score_reauth_required(reauth_required, reauth_duration) -> int:
    """
    Score reauthorization requirement. Weight: 10.
    - Not required → 10
    - Required + specified duration → 5
    - Required + unspecified duration → 3
    """
    s = str(reauth_required).strip() if reauth_required else "No"

    if s != "Yes":
        return 10

    # Required — check if duration is specified
    dur = str(reauth_duration).strip() if reauth_duration else ""
    dur_numeric = _parse_numeric(reauth_duration)
    if dur_numeric is not None:
        return 5  # Specified duration
    return 3  # Unspecified duration


def score_quantity_limits(value) -> int:
    """Score quantity limits. Weight: 5."""
    s = str(value).strip() if value else "No"
    if s in ("No", "NA", "N/A", ""):
        return 5
    return 0  # Has quantity limits


def score_specialist(value) -> int:
    """Score specialist requirement. Weight: 5."""
    s = str(value).strip() if value else "NA"
    if s in ("NA", "N/A", "No", ""):
        return 5
    return 0  # Specialist required


def score_reauth_requirements(value) -> int:
    """
    Score reauthorization requirements by restrictiveness (condition-based).
    Weight: 10.
    - NA → 10 (no requirements)
    - Response-based language → 7 (lenient)
    - Test/score-based language → 3 (restrictive)
    - Other text → 5 (moderate)
    """
    s = str(value).strip() if value else "NA"

    if s in ("NA", "N/A", "No", "", "Unspecified"):
        return 10

    lower = s.lower()

    # Restrictive: specific test scores, lab values, quantified thresholds
    restrictive_patterns = [
        r'pasi[\s\-]*\d+',
        r'bsa.*\d+\s*%',
        r'dlqi.*\d+',
        r'lab\s+value',
        r'documented.*reduction.*\d+',
        r'at\s+least\s+\d+\s*%',
        r'score.*\d+',
    ]
    for pattern in restrictive_patterns:
        if re.search(pattern, lower):
            return 3

    # Lenient: general response language
    lenient_patterns = [
        r'benefit',
        r'improvement',
        r'stabiliz',
        r'response\s+to\s+therapy',
        r'continue.*to.*respond',
        r'has\s+responded',
        r'clinical\s+response',
    ]
    for pattern in lenient_patterns:
        if re.search(pattern, lower):
            return 7

    # Default: text exists but doesn't match either pattern
    return 5


def _count_active_restrictions(params: dict, brand: str) -> int:
    """Count how many restriction categories are active (non-zero penalty)."""
    count = 0
    if str(params.get("step_through_phototherapy", "")).strip() == "Yes":
        count += 1
    if str(params.get("tb_test_required", "")).strip() in ("Yes", "Y"):
        count += 1
    if str(params.get("quantity_limits", "")).strip() == "Yes":
        count += 1
    if str(params.get("specialist_types", "")).strip() not in ("NA", "N/A", "No", ""):
        count += 1
    if str(params.get("reauth_required", "")).strip() == "Yes":
        count += 1
    age_val = str(params.get("age", "")).strip()
    if age_val not in ("No", "", "NA"):
        count += 1
    # Short auth duration counts as a restriction
    dur = _parse_numeric(params.get("initial_auth_duration"))
    if dur is not None and dur < 6:
        count += 1
    return count


def _count_real_params(params: dict) -> int:
    """Count how many extractable parameters have real (non-NA) values.

    A "real" value is anything other than NA/N/A/empty/None.
    For step_therapy_requirements, a descriptive string (>20 chars) counts
    as real even if it's not a simple "Yes"/"No".
    """
    check_keys = [
        "step_therapy_requirements", "steps_through_brands",
        "steps_through_generic", "step_through_phototherapy",
        "tb_test_required", "quantity_limits", "specialist_types",
        "initial_auth_duration", "reauth_required", "reauth_duration",
        "age", "reauth_requirements",
    ]
    real_count = 0
    for k in check_keys:
        val = str(params.get(k, "")).strip()
        if val not in ("NA", "N/A", "", "None", "No"):
            real_count += 1
        elif k == "step_therapy_requirements" and len(val) > 20:
            # Long descriptive text = real data even if not "Yes"
            real_count += 1
    return real_count


def compute_access_score(params: dict, brand: str) -> int | str:
    """
    Compute Access Score (0-100) using a layered model.

    Layer 0: Sparse data guard — very few extracted params → low score
    Layer 1: Hard floors — extreme cases that collapse the score
    Layer 2: Step-therapy base — steps determine the scoring range
    Layer 3: Secondary adjustments — other parameters add/subtract
    Layer 4: Interaction penalty — stacked restrictions compound
    Layer 5: Caps and clamp
    """
    # ── LAYER 0: SPARSE DATA GUARD ──────────────────────────────────────────
    # If almost no params have real values, the extraction likely failed.
    # Score conservatively rather than guessing.
    real_count = _count_real_params(params)
    if real_count < 2:
        return 25

    branded = _parse_numeric(params.get("steps_through_brands"))
    generic = _parse_numeric(params.get("steps_through_generic"))
    b = branded if branded is not None else 0
    g = generic if generic is not None else 0
    total_steps = b + g

    # Infer steps from step_therapy_requirements text when counts are NA.
    # If the LLM described step therapy requirements but didn't extract
    # numeric counts, assume at least 1 generic step exists.
    step_text = str(params.get("step_therapy_requirements", "")).strip()
    has_step_text = (step_text not in ("No", "NA", "N/A", "", "None", "Yes")
                     and len(step_text) > 20)
    if has_step_text and total_steps == 0:
        g = 1
        total_steps = 1

    # ── LAYER 1: HARD FLOORS ────────────────────────────────────────────────
    # Only the most extreme cases collapse to 0.
    # 4+ branded AND 3+ generic = near-impossible access
    if b >= 4 and g >= 3:
        return 0
    # 7+ total steps regardless of split
    if total_steps >= 7:
        return 0
    # 4+ branded steps alone = very restricted
    if b >= 4:
        return 25

    # ── LAYER 2: STEP-THERAPY BASE ──────────────────────────────────────────
    # Steps are the dominant factor. Any step therapy = significant barrier.
    # Calibrated against ground truth: 1G + photo + specialist ≈ 25.
    if total_steps == 0:
        base = 90
    elif total_steps == 1 and b == 0:
        base = 50  # 1 generic step — meaningful barrier
    elif total_steps == 1 and b == 1:
        base = 40  # 1 branded step — harder
    elif total_steps == 2 and b <= 1:
        base = 30  # 2 steps, at most 1 branded
    elif total_steps == 2 and b == 2:
        base = 20  # 2 branded steps — significant barrier
    elif total_steps == 3:
        base = 15
    elif total_steps == 4:
        base = 10
    else:
        base = 5

    # ── LAYER 3: SECONDARY ADJUSTMENTS ──────────────────────────────────────
    score = float(base)

    # Auth duration: longer = better access
    dur = _parse_numeric(params.get("initial_auth_duration"))
    reauth_req = str(params.get("reauth_required", "")).strip()
    if dur is not None:
        if dur >= 12:
            score += 5
        elif dur >= 6:
            score += 2
        elif dur >= 3:
            score += 0
        else:
            score -= 5  # Very short auth = penalty
    else:
        # Unspecified duration: slight negative if reauth required
        if reauth_req == "Yes":
            score -= 2
        # else: no adjustment — unspecified without reauth is neutral

    # Reauth
    reauth_req = str(params.get("reauth_required", "")).strip()
    if reauth_req == "Yes":
        reauth_dur = _parse_numeric(params.get("reauth_duration"))
        if reauth_dur is not None and reauth_dur >= 12:
            score -= 2  # Long reauth interval — mild penalty
        elif reauth_dur is not None and reauth_dur <= 3:
            score -= 8  # Frequent reauth — significant burden
        else:
            score -= 5  # Reauth required, moderate/unspecified interval

    # Reauth requirements restrictiveness
    reauth_reqs_score = score_reauth_requirements(params.get("reauth_requirements"))
    if reauth_reqs_score <= 3:
        score -= 5  # PASI/BSA/DLQI thresholds — hard to maintain
    elif reauth_reqs_score <= 5:
        score -= 3  # Moderate requirements

    # Age restriction
    age_score = score_age(params.get("age"), brand)
    if age_score <= 3:
        score -= 5  # More restrictive than FDA label

    # Binary restrictions: each is a real barrier
    if str(params.get("step_through_phototherapy", "")).strip() == "Yes":
        score -= 8  # Phototherapy is a significant requirement
    if str(params.get("tb_test_required", "")).strip() in ("Yes", "Y"):
        score -= 3
    if str(params.get("quantity_limits", "")).strip() == "Yes":
        score -= 3
    if str(params.get("specialist_types", "")).strip() not in ("NA", "N/A", "No", ""):
        score -= 5  # Specialist requirement limits access

    # ── LAYER 4: INTERACTION PENALTY ────────────────────────────────────────
    # Stacked restrictions compound
    restrictions = _count_active_restrictions(params, brand)
    if restrictions >= 6:
        score *= 0.60
    elif restrictions >= 5:
        score *= 0.70
    elif restrictions >= 4:
        score *= 0.80

    # ── LAYER 5: CAPS AND CLAMP ─────────────────────────────────────────────
    # Branded steps >= 2 caps the score regardless of other factors
    if b >= 2:
        score = min(score, 50)

    # ── LAYER 5b: BASELINE OFFSET ──────────────────────────────────────────
    # Shift scores up to reflect that most policies do grant access
    # eventually, even with restrictions.
    score += SCORE_OFFSET

    continuous = max(0, min(100, round(score)))

    # ── LAYER 6: BUCKET SNAPPING ────────────────────────────────────────────
    # Evaluation uses 5 discrete buckets. Snap to nearest.
    BUCKETS = [0, 25, 50, 75, 100]
    return min(BUCKETS, key=lambda b: abs(b - continuous))


def compute_confidence(deterministic_score, extraction: dict) -> dict:
    """Compare deterministic score with LLM's own estimate.

    Returns a dict with:
        llm_estimate: int or None
        deterministic: int
        diff: int or None
        level: "High" | "Medium" | "Low" | "N/A"
    """
    llm_raw = extraction.get("estimated_access_score")
    try:
        llm_estimate = int(llm_raw)
        llm_estimate = max(0, min(100, llm_estimate))
    except (TypeError, ValueError):
        return {
            "llm_estimate": None,
            "deterministic": deterministic_score,
            "diff": None,
            "level": "N/A",
        }

    try:
        det = int(deterministic_score)
    except (TypeError, ValueError):
        det = 0

    diff = abs(llm_estimate - det)

    if diff <= 15:
        level = "High"
    elif diff <= 30:
        level = "Medium"
    else:
        level = "Low"

    return {
        "llm_estimate": llm_estimate,
        "deterministic": det,
        "diff": diff,
        "level": level,
    }


# Valid score buckets
VALID_BUCKETS = {0, 25, 50, 75, 100}


def check_score_sanity(params: dict, score: int | str, brand: str) -> list[str]:
    """Check whether the Access Score is consistent with the extracted parameters.

    Returns a list of warning strings. Empty list = no issues.
    Deterministic, no API call.
    """
    warnings = []

    # "NA" score = too few real params, already flagged by compute_access_score
    if score == "NA":
        real = _count_real_params(params)
        warnings.append(
            f"Score=NA — only {real}/12 parameters have real values "
            f"(PDF may not cover this brand/indication)")
        return warnings

    if score not in VALID_BUCKETS:
        warnings.append(f"Score {score} is not a valid bucket ({VALID_BUCKETS})")

    b = _parse_numeric(params.get("steps_through_brands")) or 0
    g = _parse_numeric(params.get("steps_through_generic")) or 0
    total_steps = b + g

    step_therapy = str(params.get("step_therapy_requirements", "")).strip()
    reauth_req = str(params.get("reauth_required", "")).strip()
    photo = str(params.get("step_through_phototherapy", "")).strip()
    tb = str(params.get("tb_test_required", "")).strip()
    ql = str(params.get("quantity_limits", "")).strip()
    specialist = str(params.get("specialist_types", "")).strip()

    # High score but many restrictions
    if score >= 75 and total_steps >= 2:
        warnings.append(
            f"Score={score} but total steps={total_steps} "
            f"(branded={b}, generic={g}) — expected ≤50")

    # High score but branded steps cap should apply
    if score > 50 and b >= 2:
        warnings.append(
            f"Score={score} but branded_steps={b} — cap should limit to ≤50")

    # Zero/low score but no step therapy
    if score == 0 and step_therapy in ("No", "NA", "N/A", ""):
        restrictions = sum([
            photo == "Yes",
            tb in ("Yes", "Y"),
            ql == "Yes",
            specialist not in ("NA", "N/A", "No", ""),
            reauth_req == "Yes",
        ])
        if restrictions < 3:
            warnings.append(
                f"Score=0 but step_therapy={step_therapy!r} and only "
                f"{restrictions} other restriction(s) — suspiciously low")

    # No restrictions at all but score is low
    if score <= 25 and total_steps == 0 and step_therapy in ("No", "NA", "N/A", ""):
        if reauth_req != "Yes" and photo != "Yes":
            warnings.append(
                f"Score={score} but no step therapy, no reauth, "
                f"no phototherapy — expected ≥50")

    # Step therapy = Yes but 0 steps counted
    if step_therapy == "Yes" and total_steps == 0:
        warnings.append(
            f"step_therapy=Yes but branded_steps={b}, generic_steps={g} "
            f"— step counts may be wrong")

    # Step therapy = No but steps > 0
    if step_therapy in ("No", "NA", "N/A") and total_steps > 0:
        warnings.append(
            f"step_therapy={step_therapy!r} but total steps={total_steps} "
            f"— contradictory")

    # Reauth required but no duration and no requirements
    reauth_dur = str(params.get("reauth_duration", "")).strip()
    reauth_reqs = str(params.get("reauth_requirements", "")).strip()
    if reauth_req == "Yes":
        if (reauth_dur in ("NA", "N/A", "") and
                reauth_reqs in ("NA", "N/A", "")):
            warnings.append(
                "reauth_required=Yes but both duration and requirements "
                "are NA — may be a false positive")

    return warnings


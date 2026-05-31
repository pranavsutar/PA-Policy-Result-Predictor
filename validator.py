"""
validator.py — Post-processing: normalize outputs, enforce business rules,
cross-row consistency checks, brand-specific distribution validation.
"""
import re
from collections import Counter, defaultdict

from config import SUBMISSION_COLUMNS


# Internal JSON keys → submission column names
KEY_TO_COLUMN = {
    "age": "Age",
    "step_therapy_requirements": "Step Therapy Requirements Documented in Policy",
    "steps_through_brands": "Number of Steps through Brands",
    "steps_through_generic": "Number of Steps through Generic",
    "step_through_phototherapy": "Step through-Phototherapy",
    "tb_test_required": "TB Test required",
    "quantity_limits": "Quantity Limits",
    "specialist_types": "Specialist Types",
    "initial_auth_duration": "Initial Authorization Duration(in-months)",
    "reauth_duration": "Reauthorization Duration(in-months)",
    "reauth_required": "Reauthorization Required",
    "reauth_requirements": "Reauthorization Requirements Documented in Policy",
}

# Reverse mapping
COLUMN_TO_KEY = {v: k for k, v in KEY_TO_COLUMN.items()}


# Brand-specific expected distributions (from training data, filtered)
EXPECTED_DISTRIBUTIONS = {
    "STELARA": {
        "Age": {"No": 45, ">=6": 40},
        "Number of Steps through Generic": {"1": 70, "2": 20, "NA": 10},
        "Number of Steps through Brands": {"NA": 85},
        "Step through-Phototherapy": {"No": 95},
        "TB Test required": {"Yes": 55, "No": 45},
        "Initial Authorization Duration(in-months)": {"12": 65, "6": 30},
        "Reauthorization Required": {"Yes": 65, "No": 35},
        "Specialist Types": {"Dermatologist": 40, "NA": 35},
        "Quantity Limits": {"No": 85},
    },
    "TREMFYA": {
        "Age": {"No": 81, ">=18": 14},
        "Number of Steps through Generic": {"NA": 52, "1": 48},
        "Number of Steps through Brands": {"NA": 76},
        "Step through-Phototherapy": {"No": 100},
        "TB Test required": {"Yes": 52, "No": 48},
        "Initial Authorization Duration(in-months)": {"12": 57, "Unspecified": 24, "6": 19},
        "Reauthorization Required": {"Yes": 57, "No": 43},
        "Specialist Types": {"NA": 62, "Dermatologist": 33},
        "Quantity Limits": {"No": 86},
    },
}

# Cross-row consistency rules
CROSS_ROW_ALWAYS_MATCH = [
    "TB Test required",
    "Specialist Types",
    "Step through-Phototherapy",
    "Quantity Limits",
]
CROSS_ROW_NUMERIC_THRESHOLD = {
    "Initial Authorization Duration(in-months)": 6,
    "Reauthorization Duration(in-months)": 6,
}


def normalize_value(value, column_name: str) -> str:
    """Normalize a single extracted value to match expected output format.

    Value semantics (from Business Rules):
      NA  = parameter not mentioned in the document at all
      No  = document explicitly states no restriction
      Yes / Y = document explicitly states a restriction exists
    """
    if value is None:
        return "NA"

    s = str(value).strip()

    if s == "" or s.lower() == "none":
        return "NA"

    # ── TB Test required ────────────────────────────────────────────────
    # Output: "Y" (not "Yes") per Business Rules
    if column_name == "TB Test required":
        lower = s.lower()
        if lower in ("yes", "y", "true", "1", "required"):
            return "Y"
        if lower in ("no", "n", "false", "0", "not required"):
            return "No"
        if lower in ("na", "n/a", "not applicable", "not mentioned"):
            return "NA"
        return s

    # ── Reauthorization Required ────────────────────────────────────────
    # "Yes" if either reauth duration or reauth requirements is non-NA
    if column_name == "Reauthorization Required":
        lower = s.lower()
        if lower in ("yes", "y", "true", "1"):
            return "Yes"
        if lower in ("no", "n", "false", "0"):
            return "No"
        if lower in ("na", "n/a", "not applicable"):
            return "NA"
        return s

    # ── Step through-Phototherapy ───────────────────────────────────────
    # Yes = mandatory AND not in an OR statement
    # No  = not mentioned as required step
    # N/A = policy lists no criteria at all
    if column_name == "Step through-Phototherapy":
        lower = s.lower()
        if lower in ("yes", "y", "true", "1", "required"):
            return "Yes"
        if lower in ("no", "n", "false", "0", "not required"):
            return "No"
        if lower in ("na", "n/a", "not applicable", "not mentioned"):
            return "N/A"
        return s

    # ── Quantity Limits ─────────────────────────────────────────────────
    # Only what is explicitly stated as "quantity limit" — not dosage/dosing.
    # NA if not mentioned. Keep actual limit text if present.
    if column_name == "Quantity Limits":
        lower = s.lower()
        if lower in ("na", "n/a", "not applicable", "not mentioned",
                      "unspecified", "not specified", "none"):
            return "NA"
        if lower in ("no", "n", "false", "0", "no quantity limit",
                      "no quantity limits"):
            return "No"
        if lower in ("yes", "y", "true", "1"):
            return "Yes"
        # Actual limit text — keep it
        return s

    # ── Step counts ─────────────────────────────────────────────────────
    # NA if no steps required. Phototherapy excluded from counts.
    step_columns = {"Number of Steps through Brands", "Number of Steps through Generic"}
    if column_name in step_columns:
        lower = s.lower()
        if lower in ("na", "n/a", "no", "none", "0", "not applicable",
                      "not mentioned"):
            return "NA"
        try:
            n = int(float(s))
            if n == 0:
                return "NA"
            return str(n)
        except (ValueError, TypeError):
            return s

    # ── Duration fields ─────────────────────────────────────────────────
    # Number (months) or "Unspecified"
    duration_columns = {
        "Initial Authorization Duration(in-months)",
        "Reauthorization Duration(in-months)",
    }
    if column_name in duration_columns:
        lower = s.lower()
        if lower in ("na", "n/a", "not mentioned"):
            return "NA"
        if lower in ("unspecified", "not specified", "unknown"):
            return "Unspecified"
        match = re.search(r'(\d+)', s)
        if match:
            return match.group(1)
        return s

    # ── Specialist Types ────────────────────────────────────────────────
    if column_name == "Specialist Types":
        lower = s.lower()
        if lower in ("na", "n/a", "no", "none", "not specified",
                      "not mentioned"):
            return "NA"
        return s

    # ── Age ─────────────────────────────────────────────────────────────
    # Output "FDA labelled age" literally if policy says so without a number.
    # Output the actual number (e.g., ">=18") if policy specifies one.
    # Output "No" if no age restriction mentioned.
    if column_name == "Age":
        lower = s.lower()
        if lower in ("none", "not specified", "na", "n/a", "not mentioned"):
            return "No"
        if "fda" in lower:
            return "FDA labelled age"
        return s

    return s


def normalize_output(raw: dict, filename: str, brand: str) -> dict:
    """
    Normalize a raw extraction dict into submission format.
    Maps internal keys to column names, normalizes values.
    """
    row = {
        "Filename": filename,
        "Brand": brand,
    }

    for internal_key, column_name in KEY_TO_COLUMN.items():
        value = raw.get(internal_key)
        row[column_name] = normalize_value(value, column_name)

    return row


def enforce_business_rules(row: dict) -> tuple[dict, list[str]]:
    """
    Enforce business rule constraints. Returns (corrected_row, violations).
    Violations list contains strings describing what was fixed.
    """
    violations = []
    corrected = row.copy()

    reauth_req = corrected.get("Reauthorization Required", "No")
    reauth_dur = corrected.get("Reauthorization Duration(in-months)", "NA")
    reauth_reqs_text = corrected.get("Reauthorization Requirements Documented in Policy", "NA")

    # Rule 1: If Reauth Required = Yes → Reauth Duration must not be NA
    if reauth_req == "Yes" and reauth_dur == "NA":
        corrected["Reauthorization Duration(in-months)"] = "Unspecified"
        violations.append("Reauth Required=Yes but Duration=NA → set Duration=Unspecified")

    # Rule 2: If Reauth Duration is numeric OR Reauth Requirements is non-NA → Reauth Required = Yes
    dur_is_numeric = False
    try:
        int(reauth_dur)
        dur_is_numeric = True
    except (ValueError, TypeError):
        pass

    reqs_is_non_na = reauth_reqs_text not in ("NA", "N/A", "", "No")
    if (dur_is_numeric or reqs_is_non_na) and reauth_req != "Yes":
        corrected["Reauthorization Required"] = "Yes"
        violations.append(f"Reauth Duration={reauth_dur} or Requirements non-NA but Required={reauth_req} → set Required=Yes")

    # Rule 3: Steps through Brands: 0 → NA
    steps_brands = corrected.get("Number of Steps through Brands", "NA")
    if steps_brands in ("0", 0, "No"):
        corrected["Number of Steps through Brands"] = "NA"
        if steps_brands != "No":
            violations.append(f"Steps Brands={steps_brands} → set to NA")

    # Rule 4: Steps through Generic: 0 → NA
    steps_generic = corrected.get("Number of Steps through Generic", "NA")
    if steps_generic in ("0", 0, "No"):
        corrected["Number of Steps through Generic"] = "NA"
        if steps_generic != "No":
            violations.append(f"Steps Generic={steps_generic} → set to NA")

    # Rule 5: Quantity Limits — strip dosage/dosing language, keep only "quantity limit"
    ql = corrected.get("Quantity Limits", "NA")
    ql_lower = str(ql).lower()
    # If the text mentions "dosage" or "dosing" but NOT "quantity limit", it's not a QL
    if ql not in ("NA", "No", "Yes", "N/A", ""):
        has_ql_keyword = "quantity limit" in ql_lower or "quantity level limit" in ql_lower
        has_dosage_only = ("dosage" in ql_lower or "dosing" in ql_lower) and not has_ql_keyword
        if has_dosage_only:
            corrected["Quantity Limits"] = "NA"
            violations.append(f"Quantity Limits contained dosage info, not quantity limits → NA")

    # Rule 6: Access Score must be 0-100 integer (or "NA" for insufficient data)
    score = corrected.get("Access Score")
    if score is not None and str(score) != "NA":
        try:
            s = int(float(str(score)))
            s = max(0, min(100, s))
            corrected["Access Score"] = s
        except (ValueError, TypeError):
            violations.append(f"Access Score={score} is not numeric → needs manual fix")

    # Rule 6: Brand name must be uppercase
    brand = corrected.get("Brand", "")
    if brand != brand.upper():
        corrected["Brand"] = brand.upper()
        violations.append(f"Brand={brand} → {brand.upper()}")

    # Rule 7: Re-apply Rule 1 after Rule 2 corrections
    if corrected["Reauthorization Required"] == "Yes":
        if corrected.get("Reauthorization Duration(in-months)") == "NA":
            corrected["Reauthorization Duration(in-months)"] = "Unspecified"
            violations.append("Post-correction: Reauth Required=Yes but Duration still NA → Unspecified")

    return corrected, violations


def validate_cross_row_consistency(rows: list[dict]) -> list[str]:
    """
    Check consistency for PDFs that produce multiple rows (same file, different brands).
    Returns list of warning strings.
    """
    warnings = []

    # Group by filename
    by_file = defaultdict(list)
    for row in rows:
        by_file[row["Filename"]].append(row)

    for filename, file_rows in by_file.items():
        if len(file_rows) < 2:
            continue

        brands = [r["Brand"] for r in file_rows]

        # Always-match fields
        for col in CROSS_ROW_ALWAYS_MATCH:
            values = [r.get(col, "NA") for r in file_rows]
            if len(set(values)) > 1:
                warnings.append(
                    f"CROSS-ROW {filename}: {col} differs across brands "
                    f"{dict(zip(brands, values))}"
                )

        # Numeric-threshold fields
        for col, threshold in CROSS_ROW_NUMERIC_THRESHOLD.items():
            values = []
            for r in file_rows:
                v = r.get(col, "NA")
                try:
                    values.append(int(v))
                except (ValueError, TypeError):
                    values.append(None)

            numeric_values = [v for v in values if v is not None]
            if len(numeric_values) >= 2:
                diff = max(numeric_values) - min(numeric_values)
                if diff > threshold:
                    warnings.append(
                        f"CROSS-ROW {filename}: {col} differs by {diff} months "
                        f"across brands {dict(zip(brands, [r.get(col) for r in file_rows]))}"
                    )

    return warnings


def validate_distributions(rows: list[dict]) -> list[str]:
    """
    Compare output distributions against brand-specific baselines.
    Returns list of warning strings for parameters that are >25% off.
    """
    warnings = []

    # Group rows by brand
    by_brand = defaultdict(list)
    for row in rows:
        by_brand[row["Brand"]].append(row)

    for brand, brand_rows in by_brand.items():
        expected = EXPECTED_DISTRIBUTIONS.get(brand)
        if expected is None or len(brand_rows) < 5:
            continue  # Not enough rows or no baseline

        n = len(brand_rows)
        for col, expected_dist in expected.items():
            actual_counter = Counter(str(r.get(col, "NA")) for r in brand_rows)

            for value, expected_pct in expected_dist.items():
                actual_count = actual_counter.get(value, 0)
                actual_pct = (actual_count / n) * 100

                if abs(actual_pct - expected_pct) > 25:
                    warnings.append(
                        f"DISTRIBUTION {brand} / {col}: "
                        f"'{value}' is {actual_pct:.0f}% (expected ~{expected_pct}%)"
                    )

    return warnings


def validate_row_completeness(rows: list[dict]) -> list[str]:
    """Check that all required columns are present and non-empty."""
    warnings = []
    for row in rows:
        for col in SUBMISSION_COLUMNS:
            if col not in row:
                warnings.append(f"MISSING COLUMN: {row.get('Filename', '?')} / {row.get('Brand', '?')} — {col}")
            elif row[col] is None or str(row[col]).strip() == "":
                warnings.append(f"EMPTY VALUE: {row.get('Filename', '?')} / {row.get('Brand', '?')} — {col}")
    return warnings


def build_final_csv(rows: list[dict], output_path: str,
                    expected_pairs: list[tuple[str, str]] | None = None):
    """
    Write results.csv AND results.xlsx with exact column order and format.
    Validates row count and column names before writing.
    Returns list of any final warnings.
    """
    import csv

    warnings = []

    # Verify column presence
    for row in rows:
        for col in SUBMISSION_COLUMNS:
            if col not in row:
                warnings.append(f"Missing column '{col}' in row {row.get('Filename')}/{row.get('Brand')}")

    # Verify row count against expected pairs
    if expected_pairs:
        actual_pairs = {(r["Filename"], r["Brand"]) for r in rows}
        expected_set = set(expected_pairs)
        missing = expected_set - actual_pairs
        extra = actual_pairs - expected_set
        if missing:
            warnings.append(f"MISSING ROWS: {missing}")
        if extra:
            warnings.append(f"EXTRA ROWS: {extra}")

    # Build clean rows
    clean_rows = []
    for row in rows:
        clean_row = {}
        for col in SUBMISSION_COLUMNS:
            val = row.get(col, "")
            if isinstance(val, str):
                val = val.strip()
            clean_row[col] = val
        clean_rows.append(clean_row)

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in clean_rows:
            writer.writerow(row)

    # Write XLSX alongside CSV
    xlsx_path = output_path.rsplit(".", 1)[0] + ".xlsx"
    try:
        import pandas as pd
        df = pd.DataFrame(clean_rows, columns=SUBMISSION_COLUMNS)
        df.to_excel(xlsx_path, index=False, sheet_name="Results")
        print(f"  Output (xlsx): {xlsx_path}")
    except Exception as e:
        warnings.append(f"Could not write XLSX: {e}")

    return warnings

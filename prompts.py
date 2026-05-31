"""
prompts.py — System prompt, extraction prompt, validation prompt, and few-shot examples.

System prompts:
SYSTEM_PROMPT_COMPACT — compact prompt for 8b/70b extraction (12k TPM limit)
SYSTEM_PROMPT_CHUNK   — minimal prompt for chunked extraction
SYSTEM_PROMPT         — full prompt (used by per-parameter routing)
"""

# ── Compact prompt for Groq single-pass extraction ──────────────────────────
# Used for direct (non-chunked) 8b/70b extraction on Groq.
SYSTEM_PROMPT_COMPACT = """Extract PA policy parameters for {brand} (Plaque Psoriasis). Return JSON only.
VALUES: "NA"=not mentioned, "No"=no restriction, "Unspecified"=category exists but no value.
Age: "FDA labelled age" if no number, "No" if not mentioned. TB: "Y"/"No"/"NA".
Phototherapy: "Yes" only if mandatory AND not in OR. Quantity: only if labeled "quantity limit".
Steps: BRANDED=biologic/named brand, GENERIC=non-biologic. OR→least restrictive. AND across universal+specific.
JSON keys: age, step_therapy_requirements, steps_through_brands, steps_through_generic, step_through_phototherapy, tb_test_required, quantity_limits, specialist_types, initial_auth_duration, reauth_duration, reauth_required, reauth_requirements, reasoning, estimated_access_score(0-100)"""

# ── Minimal chunk prompt ────────────────────────────────────────────────────
# For chunked extraction: even shorter. Each chunk extracts what it can see;
# merge logic combines results across chunks.
SYSTEM_PROMPT_CHUNK = """Extract PA parameters for {brand} (Plaque Psoriasis) from this text chunk. Return JSON only. Use "NA" if not found in this chunk.
Keys: age, step_therapy_requirements, steps_through_brands, steps_through_generic, step_through_phototherapy, tb_test_required, quantity_limits, specialist_types, initial_auth_duration, reauth_duration, reauth_required, reauth_requirements, reasoning"""

# ── Full system prompt (used by per-parameter routing) ──────────────────────
SYSTEM_PROMPT = """You are a pharmaceutical policy analyst specializing in Prior Authorization (PA) policy extraction for Psoriasis (PsO) indications. You extract structured parameters from payer PA policy documents with precision.

KEY RULES:
1. You extract parameters ONLY for the specified target brand and the Plaque Psoriasis (PsO) indication.
2. If the policy covers multiple indications (PsA, Crohn's, UC, etc.), IGNORE all non-PsO sections.
3. If the policy covers multiple brands, focus ONLY on the target brand's PsO criteria.
4. If the policy distinguishes moderate-to-severe vs. severe PsO, use ONLY moderate-to-severe criteria.

VALUE CONVENTIONS — follow these exactly:
- "NA" = parameter is NOT MENTIONED in the document at all
- "No" = document EXPLICITLY states no restriction exists
- "Unspecified" = policy has the category but does not specify a value (e.g., has reauthorization but no duration stated)
- Numeric values for durations (e.g., 6, 12 for months) or step counts (e.g., 1, 2)

Field-specific rules:
- Age: output "FDA labelled age" literally if the policy says "FDA labelled age" or "FDA approved age" without specifying a number. Do NOT resolve to a number. Output the actual number (e.g., ">=18", ">=6") only if the policy states it. Output "No" if no age restriction is mentioned. If multiple age groups are listed, capture the youngest.
- TB Test required: output "Y" (not "Yes") if required. Output "No" if not required. Output "NA" if not mentioned.
- Step through-Phototherapy: output "Yes" ONLY if phototherapy is mandatory AND not in an OR. Output "No" if not required. Output "N/A" if policy lists no criteria at all.
- Quantity Limits: ONLY extract text explicitly labeled as "quantity limit" or "quantity level limit". Output "NA" if not mentioned. Output "No" if explicitly stated as no quantity limit. Do NOT extract dosing/dosage information.
- Reauthorization Required: output "Yes" if either reauthorization duration or reauthorization requirements are documented. Output "No" if explicitly not required. Output "NA" if not mentioned.
- Initial Authorization Duration / Reauthorization Duration: output the number of months, or "Unspecified" if the category exists but no duration is stated.

STEP COUNTING LOGIC — this is the most critical extraction:

Step 1: Identify UNIVERSAL CRITERIA
- These apply to ALL brands/indications in the document
- They may appear in a "General Guidelines" section, preamble, or preferred/non-preferred tier list
- They may NOT mention psoriasis or the target brand by name
- Example: "Non-preferred agents require trial of ONE preferred anti-TNF" = 1 branded step

Step 2: Identify INDICATION-SPECIFIC CRITERIA
- These apply specifically to PsO and/or the target brand
- Look for sections labeled "Plaque Psoriasis", "PsO", or the target brand name

Step 3: Combine via AND
- Universal criteria AND indication-specific criteria must BOTH be satisfied
- Count steps from BOTH sets

Step 4: Resolve OR conditions
- If steps appear in an OR statement, take the LEAST RESTRICTIVE path (fewer steps)
- If two statements appear with NO explicit connector between them, treat as OR

Step 5: Classify each step
- BRANDED step = requires a biologic or named brand drug (e.g., Humira, Enbrel, Cosentyx, any biosimilar)
- GENERIC step = requires a non-biologic drug (methotrexate, cyclosporine, acitretin, topicals)
- If a step mentions a drug class without naming a specific biologic, it defaults to GENERIC
- If a step mentions a drug class (e.g., "CAM antagonists") and the target drug belongs to that class, count it as BRANDED for that drug

Step 6: Phototherapy
- EXCLUDE phototherapy from branded and generic step counts
- Phototherapy is captured separately in "Step through-Phototherapy"
- Step through Phototherapy = "Yes" ONLY if phototherapy is a MANDATORY step (not in an OR condition)
- Step through Phototherapy = "No" if phototherapy is in an OR condition or not mentioned
- Step through Phototherapy = "N/A" if the policy lists no criteria at all

QUANTITY LIMITS — strict label matching:
- ONLY extract text explicitly labeled as "quantity limit" or "quantity level limit"
- Do NOT extract dosing information, dosage limits, or administration schedules
- If no explicit quantity limit label exists, output "NA"

OUTPUT FORMAT:
Return ONLY a JSON object — no markdown, no explanation outside the JSON.
Keep text fields concise:
- "step_therapy_requirements": summarize the policy criteria in 1-3 sentences, not verbatim quotes
- "reauth_requirements": summarize in 1-2 sentences
- "reasoning": 1 sentence explaining step count logic (e.g., "1 generic (MTX) AND 1 branded (preferred biologic) via OR path")

{
"age": "...",
"step_therapy_requirements": "...",
"steps_through_brands": ...,
"steps_through_generic": ...,
"step_through_phototherapy": "...",
"tb_test_required": "Y or No or NA",
"quantity_limits": "...",
"specialist_types": "...",
"initial_auth_duration": ...,
"reauth_duration": ...,
"reauth_required": "...",
"reauth_requirements": "...",
"reasoning": "...",
"estimated_access_score": 0-100
}

For numeric fields (steps_through_brands, steps_through_generic, initial_auth_duration, reauth_duration):
- Use integer values (e.g., 1, 2, 6, 12)
- Use "NA" (string) if not applicable
- Use "Unspecified" (string) if the category exists but no value is stated
- Use "No" (string) for step counts when no steps are required"""


def build_extraction_prompt(context_package) -> str:
    """Build the extraction prompt for a single (PDF, Brand) pair."""
    brand = context_package.brand
    filename = context_package.filename
    doc_type = context_package.document_type
    preferred = context_package.preferred_status

    prompt = f"""TASK: Extract PA policy parameters for {brand} for the Plaque Psoriasis (PsO) indication from the following policy document.

DOCUMENT INFO:
- Filename: {filename}
- Target Brand: {brand}
- Document Type: {doc_type}
- Preferred Status: {preferred}
- Total Pages: {context_package.total_pages}
- Pages Included Below: {context_package.relevant_pages_used}

"""

    if context_package.universal_criteria_text:
        prompt += f"""UNIVERSAL CRITERIA (applies to ALL brands — combine with indication-specific criteria via AND):
{context_package.universal_criteria_text}

"""

    prompt += f"""POLICY TEXT (relevant pages):
{context_package.full_relevant_text}

"""

    if context_package.document_type == "decision-tree":
        prompt += DECISION_TREE_INSTRUCTION + "\n"

    prompt += f"""Extract all 13 parameters for {brand} for Plaque Psoriasis. Follow the step counting logic exactly. Return JSON only."""

    return prompt


# Lean system prompt for Pass 2 — the full step-counting algorithm and output
# format instructions are unnecessary since we already have extracted JSON.
VALIDATION_SYSTEM_PROMPT = """You validate PA policy parameter extractions. Return only valid JSON. Be concise in corrections."""

VALIDATION_PROMPT_TEMPLATE = """You are validating extracted PA policy parameters. Review the extraction below against the source text and correct any errors.

TARGET: {brand} for Plaque Psoriasis (PsO)
FILENAME: {filename}

EXTRACTED VALUES:
{extracted_json}

SOURCE TEXT:
{source_text}

VALIDATION CHECKS:
1. Does each parameter match what the source text actually says?
2. Are step counts correct? (Union of universal + indication-specific, AND logic, least restrictive OR path)
3. Is the Age value correct? (youngest age group, "FDA labelled age" if no number specified, "No" if not mentioned)
4. Is TB Test correct? ("Y" if required, "No" if not required, "NA" if not mentioned)
5. Is Quantity Limits correct? (Only if explicitly labeled "quantity limit" — NOT dosing info)
6. Is Specialist Types correct? (NA if not mentioned)
7. Are authorization durations correct? (numeric months, "Unspecified" if category exists but no value)
8. Is Reauthorization Required consistent? (Yes if either reauth duration or reauth requirements is non-NA)
9. Is Step through Phototherapy correct? (Yes only if mandatory, not in OR condition)
10. Is the step therapy text complete? (includes both universal and indication-specific criteria)

BUSINESS RULE CONSTRAINTS:
- If Reauthorization Required = "Yes" → Reauthorization Duration must be numeric or "Unspecified" (NOT "NA")
- If Reauthorization Duration is numeric OR Reauthorization Requirements is non-NA → Reauthorization Required must be "Yes"
- Steps through Brands should be "NA" if no branded steps required (not "No" or 0)
- Steps through Generic should be "NA" if no generic steps required

Return ONLY the corrected JSON with the same keys — no markdown, no explanation outside JSON.
If no corrections needed, return the original JSON unchanged.
Add a "corrections" key: a short list of changes (1 line each), or null if none."""


def build_validation_prompt(brand: str, filename: str,
                           extracted_json: str, source_text: str) -> str:
    """Build the validation prompt.

    Callers are responsible for truncating source_text to fit
    within the model's token budget before calling this function.
    """
    return VALIDATION_PROMPT_TEMPLATE.format(
        brand=brand,
        filename=filename,
        extracted_json=extracted_json,
        source_text=source_text,
    )


        # Few-shot examples from the Additional Extracted Data
FEW_SHOT_EXAMPLES = [
{
"description": "TREMFYA — simple policy, 1 generic step, no branded steps",
"input_summary": "Policy requires trial of ONE formulary conventional prerequisite agent (e.g., methotrexate, cyclosporine, acitretin). TB test required. 12-month initial auth.",
"output": {
"age": "No",
"step_therapy_requirements": "ONE of the following: Patient's medication history indicates use of another biologic immunomodulator agent for the same FDA labeled indication OR Patient's medication history indicates use of ONE formulary conventional prerequisite agent for the requested indication OR Patient has an FDA labeled contraindication to at least ONE formulary conventional prerequisite agent",
"steps_through_brands": "NA",
"steps_through_generic": 1,
"step_through_phototherapy": "No",
"tb_test_required": "Y",
"quantity_limits": "NA",
"specialist_types": "NA",
"initial_auth_duration": 12,
"reauth_duration": "Unspecified",
"reauth_required": "Yes",
"reauth_requirements": "Criteria for renewal approval require ALL of the following: 1. Patient has been previously approved 2. Patient has an FDA labeled indication 3. Patient has had clinical improvement 4. Patient will NOT be using in combination with another biologic 5. Requested dose is within FDA labeled dosing",
},
},
{
"description": "STELARA — FDA labelled age, phototherapy in OR (does not count), specialist required",
"input_summary": "Policy requires 3-month trial of acitretin, methotrexate, or cyclosporine OR UVB/PUVA for 3 months. Specialist required. Age = FDA labelled age (no number specified).",
"output": {
"age": "FDA labelled age",
"step_therapy_requirements": "Patients must meet one of the following criteria: had a 3-month trial of acitretin, methotrexate, or cyclosporine therapy resulting in intolerance or clinical failure OR have tried UVB/coal tar or PUVA/topical corticosteroids for at least 3 months OR have tried and failed at least two of the following: acitretin, methotrexate, cyclosporine",
"steps_through_brands": "NA",
"steps_through_generic": 1,
"step_through_phototherapy": "No",
"tb_test_required": "No",
"quantity_limits": "NA",
"specialist_types": "Appropriate Specialist",
"initial_auth_duration": 12,
"reauth_duration": "NA",
"reauth_required": "No",
"reauth_requirements": "NA",
},
},
{
"description": "STELARA — Reference tab worked example with universal + indication-specific criteria (AND logic)",
"input_summary": "Universal criteria: must try Yesintek (1 branded step). Indication-specific: previously received biologic (1 branded) OR inadequate response to phototherapy/methotrexate/cyclosporine/acitretin (1 generic). Two statements with no connector = OR, take least restrictive = 1 generic.",
"output": {
"age": ">=6",
"step_therapy_requirements": "Documentation for all indications: The patient is unable to take Yesintek (ustekinumab-kfce), where indicated, for the given diagnosis due to a trial and inadequate treatment response or intolerance, or a contraindication. Authorization of 12 months may be granted for members 6 years of age and older who have previously received a biologic or targeted synthetic drug (e.g., Sotyktu, Otezla) indicated for treatment of moderate to severe plaque psoriasis. At least 3% of body surface area (BSA) is affected and the member meets either of the following criteria: Member has had an inadequate response or intolerance to either phototherapy (e.g., UVB, PUVA) or pharmacologic treatment with methotrexate, cyclosporine, or acitretin. Member has a clinical reason to avoid pharmacologic treatment with methotrexate, cyclosporine, and acitretin.",
"steps_through_brands": 1,
"steps_through_generic": 1,
"step_through_phototherapy": "No",
"tb_test_required": "Y",
"quantity_limits": "Quantity Level Limit: Stelara (ustekinumab) 130 mg/26 mL single-dose vial: 4 vials (1 dose). Stelara (ustekinumab) subcutaneous injection 45 mg/0.5 mL single-dose vial/prefilled syringe: 1 vial/syringe per 84 days. Exception limit: 2 vials/syringes per 28 days",
"reasoning": "Universal: 1 branded (Yesintek). Indication-specific: OR between 1 branded (biologic) and 1 generic (methotrexate/cyclosporine/acitretin). Least restrictive OR path = 1 generic. Final: 1 branded (universal) AND 1 generic (indication) = 1 branded + 1 generic.",
"specialist_types": "Dermatologist",
"initial_auth_duration": 6,
"reauth_duration": 12,
"reauth_required": "Yes",
"reauth_requirements": "Documentation of positive clinical response to therapy as evidenced by ONE of the following: reduction in body surface area (BSA) involvement from baseline OR improvement in symptoms (e.g., pruritus, inflammation) from baseline",
},
},
]


# ── Per-parameter micro-prompts for groq-optimized strategy ──────────────────
# 7 simple parameters → 8b model (~400 tokens each)
# 5 complex parameters → 70b model (combined ~1,500 tokens)

# Each micro-prompt returns a tiny JSON with just the target field(s).
# The orchestrator merges them into the full 13-field output.

MICRO_PROMPT_TB_TEST = """Extract TB test requirement for {brand} (Plaque Psoriasis).
Look for: tuberculosis, TB test, TB screen, QuantiFERON, latent TB.
Return JSON: {{"tb_test_required": "Y" or "No" or "NA"}}
- "Y" = required. "No" = explicitly not required. "NA" = not mentioned.

TEXT:
{snippet}"""

MICRO_PROMPT_REAUTH_REQUIRED = """Extract reauthorization requirement for {brand} (Plaque Psoriasis).
Look for: reauthorization, renewal, continuation criteria, re-authorization.
Return JSON: {{"reauth_required": "Yes" or "No" or "NA"}}
- "Yes" = reauth duration or requirements documented. "No" = explicitly not required. "NA" = not mentioned.

TEXT:
{snippet}"""

MICRO_PROMPT_AGE = """Extract age requirement for {brand} (Plaque Psoriasis).
Look for: years of age, >=18, >=6, pediatric, adolescent, FDA labelled age.
Return JSON: {{"age": VALUE}}
- "FDA labelled age" if referenced without number. Actual number (e.g. ">=18") if stated. "No" if not mentioned.

TEXT:
{snippet}"""

MICRO_PROMPT_INIT_AUTH = """Extract initial authorization duration for {brand} (Plaque Psoriasis).
Look for: authorization duration, approval period, approve for X months, valid for.
Return JSON: {{"initial_auth_duration": VALUE}}
- Integer months (e.g. 6, 12). "Unspecified" if category exists but no value. "NA" if not mentioned.

TEXT:
{snippet}"""

MICRO_PROMPT_REAUTH_DURATION = """Extract reauthorization duration for {brand} (Plaque Psoriasis).
Look for: reauthorization duration, renewal period, re-authorization for X months.
Return JSON: {{"reauth_duration": VALUE}}
- Integer months (e.g. 6, 12). "Unspecified" if category exists but no value. "NA" if not mentioned.

TEXT:
{snippet}"""

MICRO_PROMPT_SPECIALIST = """Extract specialist requirement for {brand} (Plaque Psoriasis).
Look for: dermatologist, specialist, prescriber type, prescribed by, in consultation with.
Return JSON: {{"specialist_types": VALUE}}
- Exact specialist name (e.g. "Dermatologist"). "NA" if not mentioned.

TEXT:
{snippet}"""

MICRO_PROMPT_QUANTITY = """Extract quantity limits for {brand} (Plaque Psoriasis).
ONLY extract text explicitly labeled "quantity limit" or "quantity level limit". NOT dosing info.
Return JSON: {{"quantity_limits": VALUE}}
- Exact limit text if labeled. "No" if explicitly no limit. "NA" if not mentioned.

TEXT:
{snippet}"""

# 8b micro-prompt registry: parameter name → template
MICRO_PROMPTS_8B = {
"tb_test_required": MICRO_PROMPT_TB_TEST,
"reauth_required": MICRO_PROMPT_REAUTH_REQUIRED,
"age": MICRO_PROMPT_AGE,
"initial_auth_duration": MICRO_PROMPT_INIT_AUTH,
"reauth_duration": MICRO_PROMPT_REAUTH_DURATION,
"specialist_types": MICRO_PROMPT_SPECIALIST,
"quantity_limits": MICRO_PROMPT_QUANTITY,
}

# Combined prompt for the 5 complex parameters → 70b model
COMPLEX_PARAMS_PROMPT = """Extract step therapy and reauthorization details for {brand} (Plaque Psoriasis). Return JSON only.

STEP COUNTING RULES:
1. Find UNIVERSAL criteria (applies to all brands, may not mention PsO)
2. Find PsO-SPECIFIC criteria for {brand}
3. Combine via AND. Within each, OR → take least restrictive path
4. BRANDED step = biologic/named brand. GENERIC step = non-biologic (MTX, cyclosporine, acitretin)
5. Phototherapy excluded from step counts, captured separately
6. No connector between statements → treat as OR

VALUES:
- step_through_phototherapy: "Yes" only if mandatory AND not in OR. "No" otherwise. "N/A" if no criteria at all.
- steps_through_brands/steps_through_generic: integer count, or "NA" if none required.
- step_therapy_requirements: 1-3 sentence summary of criteria.
- reauth_requirements: 1-2 sentence summary. "NA" if not mentioned.

Return JSON:
{{"step_through_phototherapy": "...", "step_therapy_requirements": "...", "steps_through_brands": ..., "steps_through_generic": ..., "reauth_requirements": "...", "reasoning": "1 sentence explaining step count logic"}}

DOCUMENT INFO:
- Target Brand: {brand}
- Document Type: {doc_type}
- Preferred Status: {preferred}

TEXT:
{snippet}"""

# Batched prompt for all 7 simple parameters in one 8b call
BATCHED_SIMPLE_PROMPT = """Extract these 7 PA policy parameters for {brand} (Plaque Psoriasis). Return JSON only.

RULES:
- tb_test_required: "Y" if required, "No" if not, "NA" if not mentioned
- reauth_required: "Yes" if reauth duration or requirements documented, "No" if not, "NA" if not mentioned
- age: "FDA labelled age" if referenced without number, actual number (e.g. ">=18") if stated, "No" if not mentioned
- initial_auth_duration: integer months, "Unspecified" if category exists but no value, "NA" if not mentioned
- reauth_duration: integer months, "Unspecified" if category exists but no value, "NA" if not mentioned
- specialist_types: exact name (e.g. "Dermatologist"), "NA" if not mentioned
- quantity_limits: ONLY if labeled "quantity limit" — NOT dosing info. "NA" if not mentioned

Return JSON:
{{"tb_test_required": "...", "reauth_required": "...", "age": "...", "initial_auth_duration": ..., "reauth_duration": ..., "specialist_types": "...", "quantity_limits": "..."}}

TEXT:
{snippet}"""

# Confirmation prompt for 8b: pre-analysis + source text → confirm or correct
CONFIRMATION_PROMPT = """You are verifying a pre-analysis of step therapy requirements for {brand} (Plaque Psoriasis).

{pre_analysis}

SOURCE TEXT (verify against this):
{snippet}

TASK: Confirm or correct the computed values. Also provide:
- step_therapy_requirements: 1-3 sentence summary of the criteria
- reauth_requirements: 1-2 sentence summary, or "NA" if not mentioned

Return JSON only:
{{"step_through_phototherapy": "Yes" or "No" or "N/A", "step_therapy_requirements": "...", "steps_through_brands": integer or "NA", "steps_through_generic": integer or "NA", "reauth_requirements": "...", "reasoning": "1 sentence"}}"""

CONFIRMATION_SYSTEM = "Verify pre-analyzed PA policy parameters. Correct errors. Return only valid JSON."

# System prompts for micro-extraction (kept minimal for token budget)
MICRO_SYSTEM_8B = "Extract the requested PA policy parameter. Return only valid JSON."
MICRO_SYSTEM_70B = "Extract PA policy parameters for Plaque Psoriasis. Return only valid JSON."


def build_micro_prompt(param: str, brand: str, snippet: str) -> str:
    """Build a micro-prompt for a single simple parameter (8b model).

    Callers should pre-filter to parameter-relevant sentences and
    chunk so snippet already fits within the token budget.
    """
    template = MICRO_PROMPTS_8B.get(param)
    if not template:
        raise ValueError(f"No micro-prompt for parameter: {param}")
    return template.format(brand=brand, snippet=snippet)


def build_confirmation_prompt(brand: str, pre_analysis_block: str,
                             snippet: str) -> str:
    """Build a confirmation prompt for 70b to verify pre-analyzed step therapy.

    Callers are responsible for chunking so snippet already fits
    within the token budget.
    """
    return CONFIRMATION_PROMPT.format(
        brand=brand,
        pre_analysis=pre_analysis_block,
        snippet=snippet,
    )


def build_batched_simple_prompt(brand: str, snippet: str) -> str:
    """Build a single prompt for all 7 simple parameters (8b model).

    Callers are responsible for chunking so snippet already fits
    within the token budget. No truncation applied here.
    """
    return BATCHED_SIMPLE_PROMPT.format(brand=brand, snippet=snippet)


def build_complex_prompt(brand: str, snippet: str,
                        doc_type: str = "standard",
                        preferred: str = "Unknown") -> str:
    """Build the combined prompt for 5 complex parameters (70b model).

    Callers are responsible for chunking so snippet already fits
    within the token budget. No truncation applied here.
    """
    return COMPLEX_PARAMS_PROMPT.format(
        brand=brand, snippet=snippet,
        doc_type=doc_type, preferred=preferred,
    )


DECISION_TREE_INSTRUCTION = """
DECISION-TREE FORMAT INSTRUCTIONS:
This policy uses a flowchart/decision-tree format with numbered questions
(e.g., "10. Is the diagnosis plaque psoriasis? Yes: Go to #11").
To extract parameters:
1. Start at the question that routes to plaque psoriasis (look for
"Is the diagnosis plaque psoriasis" or similar)
2. Follow the "Yes" path to find the step therapy requirements
3. Each bullet under the approval question is an AND-connected step
(e.g., "Has the patient failed to respond to each of the following:" → all are AND)
4. The approval duration is stated in the "Yes: Approve for up to X months" text
5. Follow "Go to Renewal Criteria" for reauthorization parameters
6. TB test is typically in an earlier universal question (e.g., question #4
"Has the patient been screened for tuberculosis?")
7. Preferred/non-preferred routing may appear in an earlier question
(e.g., "Is the request for a non-preferred product?")
"""


def select_few_shot_examples(brand: str, doc_type: str) -> list[dict]:
    """Select 2-3 few-shot examples matching the current row's characteristics.

    Selection priority:
    1. Always include the Reference tab worked example (has AND/OR counting logic)
    2. Add a brand-matched example
    """
    selected = []

    # Always include the Reference tab example (Example 3 — has AND/OR counting logic)
    selected.append(FEW_SHOT_EXAMPLES[2])

    # Add brand-matched example
    if brand == "TREMFYA":
        selected.append(FEW_SHOT_EXAMPLES[0])  # Simple Tremfya
    elif brand == "STELARA":
        selected.append(FEW_SHOT_EXAMPLES[1])  # FDA-age Stelara
    else:
        # For other brands, include both to show range
        selected.append(FEW_SHOT_EXAMPLES[0])
        selected.append(FEW_SHOT_EXAMPLES[1])

    return selected


def format_few_shot_examples(examples: list[dict] | None = None,
                            compact: bool = False) -> str:
    """Format few-shot examples for inclusion in the prompt.

    Args:
        compact: If True, use minimal formatting for Groq's tight token budget.
                 Omits input_summary and uses single-line JSON.
    """
    import json

    if examples is None:
        examples = FEW_SHOT_EXAMPLES

    if compact:
        # Compact format: just description + condensed JSON output
        lines = ["EXAMPLES:\n"]
        for i, ex in enumerate(examples, 1):
            output = {k: v for k, v in ex["output"].items()
                      if not k.endswith("_reasoning")}
            # Shorten long text fields for compact mode
            for key in ("step_therapy_requirements", "reauth_requirements",
                        "quantity_limits"):
                if key in output and isinstance(output[key], str) and len(output[key]) > 80:
                    output[key] = output[key][:77] + "..."
            lines.append(f"Ex{i}: {ex['description']}")
            lines.append(json.dumps(output, separators=(',', ':')))
            lines.append("")
        return "\n".join(lines)

    lines = ["EXAMPLES (for reference — follow the same output format and logic):\n"]

    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}: {ex['description']}")
        lines.append(f"Context: {ex['input_summary']}")
        # Remove reasoning keys from output for cleaner examples
        output = {k: v for k, v in ex["output"].items()
                  if not k.endswith("_reasoning")}
        lines.append(f"Output: {json.dumps(output, indent=2)}")
        lines.append("")

    return "\n".join(lines)

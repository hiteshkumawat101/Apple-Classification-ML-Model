"""Component eligibility rules for billing estimation.

Defines which components (room, doctor, dialysis, etc.) should be charged for
each type of diagnosis, and what evidence signals indicate eligibility for
conditional components.
"""

COMPONENT_RULES = {
    "room": {"type": "mandatory"},
    "doctor": {"type": "mandatory"},
    "laboratory": {"type": "mandatory"},
    "operating_theatre": {
        "type": "conditional",
        "evidence": ["ot_procedures_total", "explicit_surgery_diagnosis"],
    },
    "dialysis": {
        "type": "conditional",
        "evidence": [
            "dialysis_sessions",
            "renal_failure_diagnosis",
            "aki_diagnosis",
            "explicit_dialysis_diagnosis",
        ],
    },
    "radiology": {"type": "diagnosis_based"},
    "physiotherapy": {"type": "diagnosis_based"},
    "miscellaneous": {"type": "mandatory"},  # catch-all, never suppressed
}

# Diagnosis keywords/ICD code patterns that signal conditional component eligibility.
# Used by resolve_component_eligibility to check against diagnosis text.
EVIDENCE_KEYWORDS = {
    "explicit_surgery_diagnosis": [
        "surgery",
        "operative",
        "procedure",
        "surgical",
        "gastrectomy",
        "cholecystectomy",
        "nephrectomy",
        "hysterectomy",
        "prostatectomy",
        "mastectomy",
        "appendectomy",
        "hernia",
        "herniorrhaphy",
    ],
    "renal_failure_diagnosis": [
        "renal failure",
        "ckd",
        "chronic kidney",
        "arf",
        "acute renal",
        "esrd",
        "end-stage renal",
        "aki",
        "acute kidney",
    ],
    "aki_diagnosis": [
        "aki",
        "acute kidney",
        "acute renal",
        "arf",
    ],
    "explicit_dialysis_diagnosis": [
        "dialysis",
        "hemodialysis",
        "peritoneal dialysis",
        "hd",
        "pd",
    ],
}

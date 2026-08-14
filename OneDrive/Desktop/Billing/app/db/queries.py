import json
import logging
import re
from datetime import datetime, timezone, timedelta

import aiomysql
from app.db.connection import get_pool, get_dest_pool

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


def _normalize_room_type(room_type: str) -> str:
    """Normalize user-facing room labels to the DB's operational room key.

    UI/manual inputs often include suffixes like "Room", "Ward", punctuation,
    or extra spaces ("Deluxe Room", " deluxe-room "). The DB lookup for room
    pricing is exact-match based, so normalize these common variants first to
    avoid missing the configured rate and falling back to Gemini's raw room
    estimate.
    """
    normalized = re.sub(r"[\s_-]+", " ", (room_type or "").strip()).lower()
    normalized = re.sub(r"\b(room|ward)\b", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _room_lookup_candidates(room_type: str) -> tuple[str, str]:
    """Return both the full normalized label and the base label.

    Example:
        "Deluxe Room" -> ("deluxe room", "deluxe")

    The DB may store labels with prefixes like room numbers ("306 - Deluxe
    Room"), while the UI may send just the class name. We therefore keep both
    forms for matching.
    """
    full = re.sub(r"[\s_-]+", " ", (room_type or "").strip()).lower()
    full = re.sub(r"\s+", " ", full).strip()
    base = _normalize_room_type(room_type)
    return full, base


async def fetch_patient_by_hr_no(hr_no: str) -> dict | None:
    """Fetch current patient admission details by HR number.

    ip_admissions itself has no patient_name/gender/age/department/doctor_name/
    room_type/payment_type columns; these live in patient_profile, doctor_master
    and bed_allocations and are joined in here.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                """
                SELECT
                    ia.ip_id,
                    ia.hr_no,
                    ia.date AS admission_date,
                    ia.discharge_date,
                    ia.payment_method AS payment_type,
                    ia.admission_type,
                    pp.patient_name,
                    pp.gender,
                    pp.age,
                    dm.doctor_name,
                    dm.id AS doctor_id,
                    dm.speciality AS department,
                    (
                        SELECT ba.room_type FROM bed_allocations ba
                        WHERE ba.ip_id = ia.ip_id
                        ORDER BY ba.start_date DESC, ba.ba_id DESC LIMIT 1
                    ) AS room_type,
                    (
                        SELECT ba.room_id FROM bed_allocations ba
                        WHERE ba.ip_id = ia.ip_id
                        ORDER BY ba.start_date DESC, ba.ba_id DESC LIMIT 1
                    ) AS room_id,
                    (
                        SELECT ics.list_of_diagnosis FROM ipd_case_sheet ics
                        WHERE ics.ip_id = ia.ip_id
                        ORDER BY ics.date DESC, ics.ics_id DESC LIMIT 1
                    ) AS diagnosis
                FROM ip_admissions ia
                LEFT JOIN patient_profile pp ON pp.hr_no = ia.hr_no
                LEFT JOIN doctor_master dm
                    ON dm.id = COALESCE(NULLIF(ia.primary_doctor_id, 0), NULLIF(ia.doctor_id, 0))
                WHERE ia.hr_no = %s
                ORDER BY ia.date DESC, ia.ip_id DESC
                LIMIT 1
                """,
                (hr_no,),
            )
            return await cur.fetchone()


# accounts is the authoritative primary billing source (confirmed: its
# type='pharmacy_sales' rows are an exact mirror of the `sales` table, and its
# totals reconcile with ip_bills; ip_charges was found to significantly
# UNDERCOUNT several admissions — up to 4x low — so it must not be used as the
# bill total source). accounts.bp_id joins to bill_particular_master for the
# same category_name used elsewhere; rows without a bp_id (mainly pharmacy POS
# and lab entries) fall back to accounts.type. Non-charge rows (payments,
# concessions, adjustments) are excluded from the total.
_BILL_BUCKET_CASE = """
    CASE
        WHEN LOWER(TRIM(bpm.category_name)) = 'room rent' THEN 'room'
        WHEN LOWER(TRIM(bpm.category_name)) = 'dialysis' THEN 'dialysis'
        WHEN LOWER(TRIM(bpm.category_name)) = 'radiology' THEN 'radiology'
        WHEN LOWER(TRIM(bpm.category_name)) = 'physiotherapy' THEN 'physiotherapy'
        WHEN LOWER(TRIM(bpm.category_name)) IN ('investigations', 'lab', 'cardiac test') THEN 'laboratory'
        WHEN LOWER(TRIM(bpm.category_name)) = 'operation theatre' THEN 'operating_theatre'
        WHEN LOWER(TRIM(bpm.category_name)) IN ('consultation', 'doctors surgical fees') THEN 'doctor'
        WHEN LOWER(TRIM(bpm.category_name)) IN ('medicines', 'pharmacy') THEN 'pharmacy'
        WHEN bpm.category_name IS NOT NULL THEN 'miscellaneous'
        WHEN a.type = 'pharmacy_sales' THEN 'pharmacy'
        WHEN a.type = 'pharmacy_sales_return' THEN 'pharmacy_return'
        WHEN a.type = 'lab' THEN 'laboratory'
        WHEN a.type IN ('patient_paid', 'insurance_receipt', 'concession', 'Round Off', 'General_Voucher') THEN 'exclude'
        ELSE 'miscellaneous'
    END
"""


async def _bill_buckets(ip_id: int) -> dict:
    """Categorize an admission's charges into the 8 billing buckets, sourced
    entirely from accounts (see rationale above)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                f"""
                SELECT bucket, SUM(credit) AS total
                FROM (
                    SELECT a.credit, {_BILL_BUCKET_CASE} AS bucket
                    FROM accounts a
                    LEFT JOIN bill_particular_master bpm ON bpm.bp_id = a.bp_id
                    WHERE a.ip_id = %s AND a.credit > 0
                ) t
                GROUP BY bucket
                """,
                (ip_id,),
            )
            rows = await cur.fetchall()

    totals = {r["bucket"]: float(r["total"] or 0) for r in rows}
    pharmacy = max(totals.get("pharmacy", 0) - totals.get("pharmacy_return", 0), 0)
    return {
        "room": totals.get("room", 0),
        "doctor": totals.get("doctor", 0),
        "laboratory": totals.get("laboratory", 0),
        "radiology": totals.get("radiology", 0),
        "operating_theatre": totals.get("operating_theatre", 0),
        "dialysis": totals.get("dialysis", 0),
        "physiotherapy": totals.get("physiotherapy", 0),
        "pharmacy": pharmacy,
        "miscellaneous": totals.get("miscellaneous", 0),
    }


async def fetch_current_bills(ip_id: int) -> dict:
    """Fetch current running bills for an active admission."""
    return await _bill_buckets(ip_id)


async def fetch_admission_utilization(ip_id: int, hr_no: str) -> dict:
    """Utilization signals for the current admission (by-hr-no only): how many
    investigations, medication orders, OT procedures, dialysis sessions and
    consultant visits have occurred so far, plus the latest recorded vitals.
    Gives Gemini more signal than raw bill totals alone.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                "SELECT COUNT(*) AS total, "
                "SUM(CASE WHEN report_status = 'Completed' THEN 1 ELSE 0 END) AS completed "
                "FROM investigation WHERE ip_id = %s",
                (ip_id,),
            )
            investigations = await cur.fetchone()

            await cur.execute(
                "SELECT COUNT(*) AS total FROM ipd_medications WHERE ip_id = %s",
                (ip_id,),
            )
            medications = await cur.fetchone()

            await cur.execute(
                "SELECT COUNT(*) AS total FROM bed_allocations WHERE ip_id = %s",
                (ip_id,),
            )
            bed_transfers = await cur.fetchone()

            await cur.execute(
                "SELECT COUNT(*) AS total, "
                "SUM(CASE WHEN operation_status = 'completed' THEN 1 ELSE 0 END) AS completed "
                "FROM ot_registration WHERE ipd_no = %s",
                (ip_id,),
            )
            ot = await cur.fetchone()

            await cur.execute(
                "SELECT COUNT(*) AS total FROM dialysis_visit WHERE ip_id = %s AND program_status != 'Cancelled'",
                (ip_id,),
            )
            dialysis_sessions = await cur.fetchone()

            await cur.execute(
                "SELECT COUNT(*) AS total FROM consultant_visits WHERE ip_id = %s",
                (ip_id,),
            )
            consultant_visits = await cur.fetchone()

            await cur.execute(
                "SELECT pulse, saturation, temperature, bp1 AS systolic_bp "
                "FROM emr_vital_signs "
                "WHERE hr_no = %s AND date_time > '2000-01-01' "
                "ORDER BY date_time DESC LIMIT 1",
                (hr_no,),
            )
            latest_vitals = await cur.fetchone()

    return {
        "investigations_total": int(investigations["total"] or 0) if investigations else 0,
        "investigations_completed": int(investigations["completed"] or 0) if investigations else 0,
        "medication_orders": int(medications["total"] or 0) if medications else 0,
        "bed_transfers": int(bed_transfers["total"] or 0) if bed_transfers else 0,
        "ot_procedures_total": int(ot["total"] or 0) if ot else 0,
        "ot_procedures_completed": int(ot["completed"] or 0) if ot else 0,
        "dialysis_sessions": int(dialysis_sessions["total"] or 0) if dialysis_sessions else 0,
        "consultant_visits": int(consultant_visits["total"] or 0) if consultant_visits else 0,
        "latest_vitals": latest_vitals or {},
    }


async def fetch_room_rate(room_type: str, payment_type: str = "cash") -> float:
    """Per-day room rate from room_rate_card based on room type and payment type.

    payment_type logic:
    - "cash": Match on rate_card_name (Hospital)
    - else (insurance/corporate/government/mediclaim): Match on rate_card.description
    """
    if not room_type:
        return 0.0

    full_room_type, normalized_room_type = _room_lookup_candidates(room_type)
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            if payment_type.lower() == "cash":
                # Match on Hospital rate_card_name
                await cur.execute(
                    """
                    SELECT AVG(CAST(rrc.room_charges AS FLOAT)) AS rate
                    FROM room_rate_card rrc
                    JOIN rate_card rc ON rrc.rc_id = rc.rc_id
                    JOIN room_master rm ON rm.room_id = rrc.room_id
                    WHERE LOWER(rm.status) = 'active'
                      AND LOWER(TRIM(rc.rate_card_name)) LIKE CONCAT('%%', %s, '%%')
                      AND (
                        LOWER(TRIM(rm.room_ward_no)) = %s
                        OR LOWER(TRIM(rm.room_type)) = %s
                        OR REPLACE(REPLACE(LOWER(TRIM(rm.room_ward_no)), ' room', ''), ' ward', '') = %s
                        OR REPLACE(REPLACE(LOWER(TRIM(rm.room_type)), ' room', ''), ' ward', '') = %s
                        OR LOWER(TRIM(rm.room_ward_no)) LIKE CONCAT('%%', %s, '%%')
                        OR LOWER(TRIM(rm.room_type)) LIKE CONCAT('%%', %s, '%%')
                      )
                    """,
                    (
                        "hospital",
                        room_type.strip().lower(),
                        room_type.strip().lower(),
                        normalized_room_type,
                        normalized_room_type,
                        full_room_type,
                        full_room_type,
                    ),
                )
            else:
                # Match on description for insurance/corporate/government/mediclaim
                await cur.execute(
                    """
                    SELECT AVG(CAST(rrc.room_charges AS FLOAT)) AS rate
                    FROM room_rate_card rrc
                    JOIN rate_card rc ON rrc.rc_id = rc.rc_id
                    JOIN room_master rm ON rm.room_id = rrc.room_id
                    WHERE LOWER(rm.status) = 'active'
                      AND LOWER(TRIM(rc.description)) LIKE CONCAT('%%', %s, '%%')
                      AND (
                        LOWER(TRIM(rm.room_ward_no)) = %s
                        OR LOWER(TRIM(rm.room_type)) = %s
                        OR REPLACE(REPLACE(LOWER(TRIM(rm.room_ward_no)), ' room', ''), ' ward', '') = %s
                        OR REPLACE(REPLACE(LOWER(TRIM(rm.room_type)), ' room', ''), ' ward', '') = %s
                        OR LOWER(TRIM(rm.room_ward_no)) LIKE CONCAT('%%', %s, '%%')
                        OR LOWER(TRIM(rm.room_type)) LIKE CONCAT('%%', %s, '%%')
                      )
                    """,
                    (
                        payment_type.strip(),
                        room_type.strip().lower(),
                        room_type.strip().lower(),
                        normalized_room_type,
                        normalized_room_type,
                        full_room_type,
                        full_room_type,
                    ),
                )
            row = await cur.fetchone()
            return float(row["rate"]) if row and row["rate"] is not None else 0.0


async def fetch_doctor_round_rate(doctor_name: str) -> float:
    """Per-round (IPD) fee for a doctor, used as the daily doctor-round rate."""
    if not doctor_name:
        return 0.0
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                """
                SELECT hospital_fees FROM doctor_master
                WHERE LOWER(doctor_name) = LOWER(%s)
                LIMIT 1
                """,
                (doctor_name,),
            )
            row = await cur.fetchone()
            return float(row["hospital_fees"]) if row and row["hospital_fees"] is not None else 0.0


async def fetch_utilization_intensity(ip_id: int) -> float:
    """Cheap complexity/severity proxy independent of diagnosis text: investigation
    count + medication order count + consultant visits + bed transfers +
    3x OT procedures + 2x dialysis sessions. Used to scale the anchor's
    per-day rate for patients busier or quieter than their diagnosis-matched
    peers, since diagnosis text alone (esp. a bare comorbidity list with no
    primary condition) often carries no cost signal.

    consultant_visits and bed_transfers were added after a correlation check
    against 500+ real discharged admissions' total bills: consultant_visits
    was the SINGLE strongest correlate found (r=+0.82, ahead of every
    component already in this formula) and bed_transfers was close behind
    OT (r=+0.72) — both were already fetched for the prompt's utilization
    text but never fed into this numeric ratio. Weighted at 1x, the same
    tier as investigations/medications, since neither showed the outsized
    per-event cost that justifies OT's/dialysis's higher weights.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM investigation WHERE ip_id = %s) AS investigations,
                    (SELECT COUNT(*) FROM ipd_medications WHERE ip_id = %s) AS medications,
                    (SELECT COUNT(*) FROM ot_registration WHERE ipd_no = %s) AS ot,
                    (SELECT COUNT(*) FROM dialysis_visit WHERE ip_id = %s AND program_status != 'Cancelled') AS dialysis,
                    (SELECT COUNT(*) FROM consultant_visits WHERE ip_id = %s) AS consultant_visits,
                    (SELECT COUNT(*) FROM bed_allocations WHERE ip_id = %s) AS bed_transfers
                """,
                (ip_id, ip_id, ip_id, ip_id, ip_id, ip_id),
            )
            row = await cur.fetchone()
    return (
        float(row["investigations"] or 0)
        + float(row["medications"] or 0)
        + float(row["consultant_visits"] or 0)
        + float(row["bed_transfers"] or 0)
        + 3 * float(row["ot"] or 0)
        + 2 * float(row["dialysis"] or 0)
    )


# Common words that pass the ">3 chars" keyword-length filter but carry no
# diagnostic meaning; left in, they turn the keyword-overlap hard filter into
# a near-match-anything filter (e.g. "with" matched most of discharge_summary,
# starving "Coronary Angioplasty with Stenting" of any real angioplasty cases).
_KEYWORD_STOPWORDS = {
    "with", "from", "also", "have", "been", "were", "having", "after", "before",
    "under", "done", "this", "that", "into", "onto", "upon", "over", "then",
    "than", "when", "where", "which", "while", "during", "each", "both",
    "such", "only", "same", "very", "your", "their", "these", "those", "there",
}

# Generic surgical/procedure verbs that are rare enough overall to score high
# on the IDF weight in _keyword_weight() (so the stopword+frequency filters
# don't catch them) but are used across unrelated specialties — e.g. "grafting"
# matched SKIN grafting in orthopedic/plastic wound-care cases for a query
# meant to find CORONARY ARTERY BYPASS GRAFTING, starving the match of any
# actual cardiac case and pulling the whole candidate pool (and its cost
# anchor) toward unrelated, sometimes much more expensive, trauma cases.
_GENERIC_PROCEDURE_TERMS = {
    "grafting", "repair", "revision", "reconstruction", "excision",
    "removal", "resection", "debridement", "exploration", "release",
}


async def _keyword_weight(cur, keyword: str, total_cases: int) -> float:
    """IDF-style weight: a keyword matching a huge fraction of all discharge
    records (e.g. "HYPERTENSION", "DIABETES" — background comorbidities
    listed on nearly every diagnosis) is far less discriminative than one
    matching a small fraction. Backtesting found the stopword list alone
    insufficient — common CLINICAL terms aren't generic English words but
    are just as non-diagnostic of a specific admission's cost profile.
    """
    await cur.execute(
        "SELECT COUNT(*) AS c FROM discharge_summary WHERE diagnosis LIKE %s OR procedure_done LIKE %s",
        (f"%{keyword}%", f"%{keyword}%"),
    )
    row = await cur.fetchone()
    freq = (row["c"] or 0) / total_cases if total_cases else 0
    if freq < 0.02:
        return 5.0
    elif freq < 0.10:
        return 2.0
    return 0.3


async def _emr_diagnosis_hr_no_indexed() -> bool:
    """Whether emr_diagnosis.hr_no has an index. Without one, the ICD-code
    matching JOIN forces a full table scan of emr_diagnosis per candidate
    admission — measured 50s+ on real data even after rewriting the query as
    a single aggregating JOIN, confirming it's a missing-index problem, not a
    query-shape one. Only SELECT access to source_db is available, so this
    can only be detected, not fixed, here — CREATE INDEX idx_emr_diagnosis_hr_no
    ON emr_diagnosis(hr_no) needs to be run by whoever administers that DB.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                """
                SELECT COUNT(*) AS c FROM information_schema.statistics
                WHERE table_schema = DATABASE() AND table_name = 'emr_diagnosis' AND column_name = 'hr_no'
                """
            )
            row = await cur.fetchone()
    return bool(row and row["c"])


async def fetch_admission_icd_codes(hr_no, admission_date, discharge_date=None) -> list[str]:
    """ICD codes for a SPECIFIC admission. emr_diagnosis is patient-level —
    it accumulates across a patient's entire history, not scoped to one
    admission (confirmed on real data: a patient with 3 admissions spanning
    2022-2026 had all 3 admissions' diagnoses mixed in one list). Windowing
    to entries created around this admission's own dates avoids pulling in
    diagnoses from unrelated earlier/later visits by the same patient.
    """
    if not hr_no or not admission_date:
        return []
    window_start = admission_date - timedelta(days=30)
    if discharge_date and str(discharge_date) != "0000-00-00":
        window_end = discharge_date + timedelta(days=3)
    else:
        window_end = datetime.now(IST).date() + timedelta(days=3)

    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                """
                SELECT DISTINCT dm.icd_code
                FROM emr_diagnosis ed
                JOIN diagnosis_master dm ON dm.diagnosis_id = ed.diagnosis_id
                WHERE ed.hr_no = %s AND ed.created_date_time BETWEEN %s AND %s
                  AND dm.icd_code IS NOT NULL AND dm.icd_code != ''
                """,
                (hr_no, window_start, window_end),
            )
            rows = await cur.fetchall()
    return [r["icd_code"] for r in rows]


async def _icd_code_weight(cur, icd_code: str, total_hr: int) -> float:
    """Same IDF-style rarity weighting as _keyword_weight, applied to exact
    ICD codes instead of free-text substrings — precise (no false-positive
    substring overlaps) but still needs rarity weighting since a code like
    I10 (hypertension) is as non-discriminative as the word ever was.
    """
    await cur.execute(
        """
        SELECT COUNT(DISTINCT ed.hr_no) AS c
        FROM emr_diagnosis ed
        JOIN diagnosis_master dm ON dm.diagnosis_id = ed.diagnosis_id
        WHERE dm.icd_code = %s
        """,
        (icd_code,),
    )
    row = await cur.fetchone()
    freq = (row["c"] or 0) / total_hr if total_hr else 0
    if freq < 0.02:
        return 5.0
    elif freq < 0.10:
        return 2.0
    return 0.3


# Strips leading/trailing punctuation only (not interior hyphens, so
# "post-operative" stays intact) -- a bare .strip() leaves a comma/semicolon
# glued to a token in a comorbidity list ("Diabetes Mellitus, Hypertension,
# Acute Pancreatitis" -> "Mellitus,", "Hypertension,"), which then requires
# that literal punctuation to appear in the same spot in a candidate's
# discharge text to ever match -- confirmed on real data to silently kill
# every multi-word condition's last keyword, and to make
# fetch_diagnosis_component_support's AND-matching (needs every keyword to
# match) fail outright on any comma/semicolon-separated diagnosis text.
_KEYWORD_PUNCT_RE = re.compile(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$")


def _extract_diagnosis_keywords(text: str) -> list[str]:
    return [
        kw for kw in (_KEYWORD_PUNCT_RE.sub("", raw) for raw in (text or "").split())
        if len(kw) > 3
        and kw.lower() not in _KEYWORD_STOPWORDS
        and kw.lower() not in _GENERIC_PROCEDURE_TERMS
    ][:6]


_DIAGNOSIS_SEGMENT_RE = re.compile(r"[,;]")


def _extract_diagnosis_segments(text: str) -> list[list[str]]:
    """Split diagnosis/procedure text on comma/semicolon into segments and
    extract keywords from each independently, instead of pooling every word
    across the whole string into one flat list.

    A comma-separated diagnosis list ("Acute Myocardial Infarction, Diabetes,
    Hypertension, Dyslipidemia") mixes words describing ONE condition
    ("Myocardial", "Infarction") with words naming SEPARATE, unrelated
    comorbidities. Picking the N rarest WORDS overall (regardless of which
    segment they came from) isn't enough: "Dyslipidemia" can itself be rare
    enough in the discharge-summary text to pass the same specificity bar as
    "Myocardial"/"Infarction" while still naming a DIFFERENT diagnosis --
    confirmed on real data, gating on it alongside AMI pulled in 17 extra
    dyslipidemia-flagged, non-AMI peers under the same doctor, dragging the
    anchor to roughly half the true AMI-only median. Keeping segment
    boundaries lets _select_gating_keywords prefer the single most specific
    SEGMENT instead of the single most specific WORD regardless of which
    unrelated condition it happened to come from.
    """
    return [_extract_diagnosis_keywords(segment) for segment in _DIAGNOSIS_SEGMENT_RE.split(text or "")]


# Shared between find_similar_cases (OR-matching, wide recall) and
# fetch_diagnosis_component_support (AND-matching, precision) -- both need to
# tell a genuinely specific diagnosis term apart from a common comorbidity
# word before using it to gate matching, not just to score/weight it.
GATING_MAX_KEYWORDS = 4
GATING_MIN_WEIGHT = 1.0


async def _select_gating_keywords(
    cur, segments: list[list[str]], total_cases: int, max_keywords: int = GATING_MAX_KEYWORDS,
) -> tuple[list[str], dict[str, float]]:
    """Pick the gating subset from the single most specific SEGMENT (see
    _extract_diagnosis_segments), not the single most specific WORDS pooled
    across every segment -- returns (gating subset, weight-by-keyword map
    covering ALL keywords in ALL segments, for scoring/tie-breaking).

    Only keywords ABOVE the common-word weight tier (see _keyword_weight: 0.3
    = matches >=10% of all discharge records) can win a segment or gate
    matching. A common comorbidity word like "Diabetes" (weight 0.3, 28% of
    records) or "Hypertension" (0.3, 39%) was observed independently pulling
    in candidates that merely mention it in an unrelated admission's
    discharge text, swinging a real peer-derived anchor ~26% purely from pool
    composition. A fixed top-N cap alone doesn't fix this when there are only
    a few keywords total (e.g. 5 words -> capping to 4 still lets most common
    terms through just to fill the quota), hence the weight-tier filter.

    Segmenting matters because a common-word filter alone still isn't enough:
    a word can be genuinely RARE in the discharge-summary text while still
    naming a totally different condition than the primary one (e.g.
    "Dyslipidemia" in "Acute Myocardial Infarction, Diabetes, Hypertension,
    Dyslipidemia" passed the same specificity bar as "Myocardial"/
    "Infarction") -- gating on it too pulled in 17 extra dyslipidemia-flagged,
    non-AMI peers under the same doctor on real data, roughly halving the
    anchor from the true AMI-only figure. Scoring each segment by its own
    MAX keyword weight and gating on only the winning segment's keywords
    keeps a multi-word condition ("Myocardial" + "Infarction") gating
    together while a separately-listed, comparably-rare-but-unrelated
    comorbidity can't independently widen the pool.

    Returns an EMPTY gating list (not a fallback to all keywords) when every
    segment is entirely common words -- confirmed on real data that gating on
    a single common word alone (e.g. diagnosis="Acute", weight 0.3, 27% of
    all records) still returns 10 "matched" peers dominated by doctor/
    department ranking, marked diagnosis_matched=True even though the "match"
    carries no real diagnostic signal. Callers must treat an empty return the
    same as no diagnosis signal at all.

    max_keywords lets a caller narrow further than the default 4 --
    fetch_diagnosis_component_support passes 1: even after excluding common
    terms and unrelated segments, requiring TWO independently-specific terms
    within the SAME segment to co-occur (e.g. "Pancreatitis" AND "Mellitus"
    if a primary segment ever contained both) could still collapse a sample
    below the 10-case floor, for a check whose actual purpose is ruling out
    GENERIC-term contamination, not demanding every specific word match.
    """
    all_keywords = [kw for segment in segments for kw in segment]
    weighted = [(kw, await _keyword_weight(cur, kw, total_cases)) for kw in all_keywords]
    weight_by_keyword = dict(weighted)

    non_empty_segments = [segment for segment in segments if segment]
    if not non_empty_segments:
        return [], weight_by_keyword

    def segment_specificity(segment: list[str]) -> float:
        return max(weight_by_keyword[kw] for kw in segment)

    winning_segment = max(non_empty_segments, key=segment_specificity)
    if segment_specificity(winning_segment) <= GATING_MIN_WEIGHT:
        return [], weight_by_keyword

    specific = sorted(
        (kw for kw in winning_segment if weight_by_keyword[kw] > GATING_MIN_WEIGHT),
        key=lambda kw: weight_by_keyword[kw], reverse=True,
    )
    gating_keywords = specific[:max_keywords]
    return gating_keywords, weight_by_keyword


# Mirrors estimator.BILL_KEYS (component_bills buckets, pharmacy excluded --
# it's a separate top-level field in EstimateResponse, not a component).
# Not imported from estimator.py to avoid a circular import (estimator.py
# imports from this module).
_COMPONENT_BUCKETS = [
    "room", "doctor", "laboratory", "radiology",
    "operating_theatre", "dialysis", "physiotherapy", "miscellaneous",
]

COMPONENT_SUPPORT_SAMPLE_LIMIT = 150

# A component counts as "supported" for a diagnosis only if at least this
# fraction of genuinely diagnosis-matched historical cases carried a charge
# for it. 30% threshold filters out cases where 70%+ of patients don't need
# the component (e.g., only 25% of TKR cases have dialysis due to comorbidities,
# not the procedure itself). This lets Gemini see the similar cases and make
# intelligent decisions based on frequency, without needing hardcoded rules.
COMPONENT_SUPPORT_RATE_THRESHOLD = 0.30

# Below this many matched cases, a rate is too noisy to act on (e.g. 0/3 looks
# like "zero support" but is just a small sample) — treat as unknown instead.
COMPONENT_SUPPORT_MIN_SAMPLE = 10


async def fetch_diagnosis_component_support(procedure_or_diagnosis: str) -> dict[str, bool] | None:
    """Whether historical cases whose diagnosis/procedure TEXT matches this one
    (regardless of doctor/department) carried a charge in each billing bucket
    often enough (>= COMPONENT_SUPPORT_RATE_THRESHOLD) to be considered
    "supported" — used to suppress a bucket in the final prediction when a
    diagnosis has essentially no historical support for it, independent of
    what the doctor/department-ranked similar_cases pool happens to contain.

    find_similar_cases's peer pool is not enough for this: when the diagnosis
    signal is weak, it deliberately boosts doctor/department matching to
    compensate, which can rank in cases with a genuinely unrelated diagnosis
    (e.g. a same-doctor sepsis case surfacing for an appendicitis query) —
    observed carrying a dialysis charge that then leaked into an appendicitis
    prediction, a diagnosis that never involves dialysis. This checks a wider,
    diagnosis-ONLY sample (no doctor/department boost, no LIMIT to a top-N
    ranked pool) so an isolated contaminated peer can't manufacture "support"
    for a component the diagnosis never actually needs.

    Returns None (meaning "unknown, don't suppress anything") when the
    diagnosis text yields no usable keywords, or fewer than
    COMPONENT_SUPPORT_MIN_SAMPLE historical cases match them — absence of
    data is not evidence of absence of need.

    Requires ALL GATING keywords to match (AND), not ANY (OR) like
    find_similar_cases uses for its wide-recall candidate pool — tested
    directly against real data: "Acute Appendicitis" under OR-matching pulled
    in every "Acute X" case (Acute Renal Failure, Acute MI, ...) and reported
    dialysis as "supported" purely because some UNRELATED "Acute" diagnosis
    needed it, the exact contamination this function exists to rule out.
    AND-matching on the specific/rare keywords only (e.g. requiring
    "Appendicitis" specifically, not just "Acute") keeps this sample precise
    at the cost of recall, which is the right tradeoff here: a false negative
    just misses suppressing a rare real cost, a false positive lets the
    original contamination back in. Common comorbidity words are excluded from
    the AND-clause the same way find_similar_cases excludes them from gating
    (see _select_gating_keywords) — requiring EVERY comorbidity word in a
    list like "Diabetes Mellitus, Hypertension, Acute Pancreatitis" to
    co-occur verbatim collapsed a 33-case Pancreatitis sample to under the
    10-case floor on real data, disabling this check outright for any
    comorbidity-heavy diagnosis text, which is precisely the input this
    safety check is most needed for.
    """
    segments = _extract_diagnosis_segments(procedure_or_diagnosis)
    if not any(segments):
        return None

    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute("SELECT COUNT(*) AS c FROM discharge_summary")
            total_row = await cur.fetchone()
            total_cases = total_row["c"] if total_row else 0
            gating_keywords, _ = await _select_gating_keywords(cur, segments, total_cases, max_keywords=1)
            if not gating_keywords:
                # Every extracted keyword was common/generic (e.g. "Acute"
                # alone) -- no term specific enough to check support against,
                # same "unknown, don't suppress anything" fallback as no
                # usable keywords at all. Building the AND-clause anyway would
                # produce an empty "AND ()" and a SQL syntax error.
                return None

            like_clauses = " AND ".join(["(ds.diagnosis LIKE %s OR ds.procedure_done LIKE %s)"] * len(gating_keywords))
            like_params = []
            for kw in gating_keywords:
                like_params.extend([f"%{kw}%", f"%{kw}%"])

            await cur.execute(
                f"""
                SELECT ia.ip_id FROM ip_admissions ia
                JOIN discharge_summary ds ON ds.ip_id = ia.ip_id
                WHERE ia.discharge_date IS NOT NULL AND ia.discharge_date != '0000-00-00'
                  AND ({like_clauses})
                ORDER BY ia.discharge_date DESC
                LIMIT %s
                """,
                like_params + [COMPONENT_SUPPORT_SAMPLE_LIMIT],
            )
            ip_ids = [r["ip_id"] for r in await cur.fetchall()]
            if len(ip_ids) < COMPONENT_SUPPORT_MIN_SAMPLE:
                return None

            fmt = ",".join(["%s"] * len(ip_ids))
            await cur.execute(
                f"""
                SELECT ip_id, bucket, SUM(credit) AS total
                FROM (
                    SELECT a.ip_id, a.credit, {_BILL_BUCKET_CASE} AS bucket
                    FROM accounts a
                    LEFT JOIN bill_particular_master bpm ON bpm.bp_id = a.bp_id
                    WHERE a.credit > 0 AND a.ip_id IN ({fmt})
                ) t
                GROUP BY ip_id, bucket
                """,
                tuple(ip_ids),
            )
            rows = await cur.fetchall()

    charged_ip_ids = {bucket: set() for bucket in _COMPONENT_BUCKETS}
    for row in rows:
        bucket = row["bucket"]
        if bucket in charged_ip_ids and float(row["total"] or 0) > 0:
            charged_ip_ids[bucket].add(row["ip_id"])

    sample_size = len(ip_ids)
    return {
        bucket: (len(ids_with) / sample_size) >= COMPONENT_SUPPORT_RATE_THRESHOLD
        for bucket, ids_with in charged_ip_ids.items()
    }


_MULTI_VALUE_RE = re.compile(r"[,;/]")


def _split_multi_value(value: str) -> list[str]:
    """Split a comma/semicolon/slash-separated field -- a dual-specialty
    doctor's department might be sent as "Cardiology, Nephrology" or
    "Cardiology/Nephrology" -- into individual values to match against ANY
    of, instead of requiring the whole compound string to match verbatim
    (which never matches anything: doctor_master.speciality is never
    literally stored as a comma-joined compound). A single value with no
    separator returns unchanged as a 1-element list.
    """
    return [v.strip() for v in _MULTI_VALUE_RE.split(value or "") if v.strip()]


def _department_stem(department: str) -> str:
    """First ~6 alpha characters of a department name, for a fuzzy LIKE match
    against doctor_master.speciality free text. speciality is written as a
    full descriptive phrase (e.g. "Traumatology, Joint Replacement & General
    Orthopaedics" for a doctor whose input department is "Orthopaedics",
    "General, Laparoscopic and Podiatric Surgeon" for one whose input
    department is "Surgery") rather than the app's standardized department
    names, so only a short substring stem reliably bridges the two --
    confirmed on real data: an EXACT match against speciality scores zero
    for every doctor checked, silently discarding this ranking signal
    entirely regardless of what department value is passed.
    """
    first_word = "".join(ch for ch in department.split()[0] if ch.isalpha()) if department.split() else ""
    return first_word[:6]


async def find_similar_cases(patient: dict, limit: int = 10) -> list[dict]:
    """Find similar discharged cases, ranked by match quality.

    Diagnosis/procedure keyword overlap is a hard filter (a case must match at
    least one keyword to be considered at all). Doctor, department and room
    type are NOT hard filters — they're ranking signals, weighted highest to
    lowest in that order. Hard-filtering on department (as before) starved the
    candidate pool for less common diagnoses (as few as 3 matches), which fed
    the LLM too few reference points; ranking instead of filtering keeps the
    pool wide while still surfacing the closest matches first.
    """
    pool = await get_pool()
    procedure = patient.get("procedure_done") or patient.get("diagnosis") or ""
    department = patient.get("department") or ""
    room_type = patient.get("room_type") or ""
    doctor_id = patient.get("doctor_id")
    doctor_name = patient.get("doctor_name") or ""
    admission_type = patient.get("admission_type") or ""

    # ICD codes (admission-window-scoped — see fetch_admission_icd_codes) are
    # the primary matching signal when available: exact-code matching avoids
    # the false-positive risk of free-text LIKE substring overlaps. Gated on
    # emr_diagnosis.hr_no being indexed — without it, the join scans all
    # 35k+ rows per candidate admission (measured 50s+ on real data). Checked
    # dynamically so this activates automatically once the index is added,
    # with no code change needed. Free-text keywords are the fallback both
    # when there's no emr_diagnosis coverage AND while the index is missing.
    icd_codes = []
    if await _emr_diagnosis_hr_no_indexed():
        icd_codes = await fetch_admission_icd_codes(
            patient.get("hr_no"), patient.get("admission_date"), patient.get("discharge_date")
        )
    segments = [] if icd_codes else _extract_diagnosis_segments(procedure)
    keywords = [kw for segment in segments for kw in segment]

    conditions = ["ia.discharge_date IS NOT NULL", "ia.discharge_date != '0000-00-00'"]
    where_params = []
    score_terms = []
    score_params = []
    signal_weights = []
    gating_keywords = []

    use_icd_path = bool(icd_codes)
    if icd_codes:
        # Cap to the rarest 4 codes: fewer codes to match keeps the JOIN
        # tight and the candidate pool small. common codes contribute little
        # weight anyway.
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute("SELECT COUNT(DISTINCT hr_no) AS c FROM ip_admissions")
                total_row = await cur.fetchone()
                total_hr = total_row["c"] if total_row else 0
                weighted_codes = [(code, await _icd_code_weight(cur, code, total_hr)) for code in icd_codes]
        weighted_codes.sort(key=lambda x: x[1], reverse=True)
        weighted_codes = weighted_codes[:4]
        icd_codes = [code for code, _ in weighted_codes]
        signal_weights.extend(w for _, w in weighted_codes)

    elif keywords:
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute("SELECT COUNT(*) AS c FROM discharge_summary")
                total_row = await cur.fetchone()
                total_cases = total_row["c"] if total_row else 0
                gating_keywords, weight_by_keyword = await _select_gating_keywords(cur, segments, total_cases)

        if gating_keywords:
            # avg_signal_weight (drives weak_diagnosis_signal / the doctor-
            # department boost below) reflects the GATING keywords' specificity
            # only, not diluted by common comorbidity terms excluded from
            # gating -- otherwise a single genuinely specific term (e.g.
            # "Pancreatitis", weight 5.0) gets its own average dragged down
            # just by ALSO listing common comorbidities, wrongly triggering
            # the boost and reranking the SAME candidate pool differently
            # depending on which comorbidities were listed.
            signal_weights.extend(weight_by_keyword[kw] for kw in gating_keywords)

            like_clauses = " OR ".join(["(ds.diagnosis LIKE %s OR ds.procedure_done LIKE %s)"] * len(gating_keywords))
            conditions.append(f"({like_clauses})")
            for kw in gating_keywords:
                where_params.extend([f"%{kw}%", f"%{kw}%"])
        else:
            # EVERY extracted keyword is common (e.g. diagnosis="Acute" alone,
            # weight 0.3, 27% of all records) -- confirmed on real data that
            # gating on it anyway still returns 10 "matched" peers dominated
            # by doctor/department ranking, marked diagnosis_matched=True even
            # though the "match" carries no real diagnostic signal. Dropping
            # the keyword filter entirely here (leaving signal_weights empty,
            # which drives weak_diagnosis_signal=True below) and leaving
            # has_diagnosis_signal False (see below) treats this exactly like
            # a genuine no-diagnosis-match: doctor/department/room ranking
            # only, correctly downgraded confidence.
            logger.info(
                "find_similar_cases: every extracted keyword %s is common/generic (none specific "
                "enough to gate on) -> dropping the keyword filter, doctor/department/room only",
                keywords,
            )

        # Only the GATING keywords (the winning segment) score -- confirmed on
        # real data that scoring non-gating keywords too (e.g. "Diabetes",
        # "Dyslipidemia" alongside a Coronary Artery Bypass Grafting primary
        # segment) still swaps out which candidates land inside the LIMIT
        # cutoff within a large match_score-tied pool (8 of 25 CABG peers
        # differed with vs. without comorbidities appended, purely from this
        # scoring nudge), even though they can no longer WIDEN the pool via
        # gating. Comorbidity text is informational context for Gemini's
        # prompt, not a peer-matching signal, once it's lost the gating vote.
        for kw in gating_keywords:
            weight = weight_by_keyword[kw]
            score_terms.append(f"(CASE WHEN ds.diagnosis LIKE %s THEN {weight} ELSE 0 END)")
            score_params.append(f"%{kw}%")
            score_terms.append(f"(CASE WHEN ds.procedure_done LIKE %s THEN {weight * 0.7} ELSE 0 END)")
            score_params.append(f"%{kw}%")

    if patient.get("ip_id"):
        conditions.append("ia.ip_id != %s")
        where_params.append(patient["ip_id"])

    where_clause = " AND ".join(conditions)

    # When the diagnosis signal is weak (low avg weight — nothing rare/specific
    # found, ICD-coded or free-text), diagnosis alone can't tell similar cases
    # apart. Lean harder on doctor/department/admission_type instead of
    # treating them as minor tiebreakers, since a given doctor in a given
    # department tends to see a consistent case mix regardless of exactly how
    # the diagnosis was coded or phrased.
    avg_signal_weight = sum(signal_weights) / len(signal_weights) if signal_weights else 0
    weak_diagnosis_signal = avg_signal_weight < 1.5
    boost = 2.5 if weak_diagnosis_signal else 1.0

    # Under GROUP BY (the ICD aggregating path), these per-admission-constant
    # terms must be wrapped in MAX() even though they don't vary within a
    # group — MySQL's ONLY_FULL_GROUP_BY mode (not currently set on this
    # server, but not guaranteed) rejects non-aggregated columns otherwise.
    wrap = (lambda expr: f"MAX({expr})") if use_icd_path else (lambda expr: expr)

    if doctor_id:
        score_terms.append(wrap(f"(CASE WHEN dm.id = %s THEN {10 * boost} ELSE 0 END)"))
        score_params.append(doctor_id)
    elif doctor_name:
        # Substring match, not equality: confirmed on real data that callers
        # sending a short name ("Dr. Ranjiith") against the full
        # doctor_master record ("Dr. Ranjiith G. Dhakshina") score ZERO under
        # exact equality even though it's the same doctor -- this is the
        # highest-weighted ranking signal (10x boost) and was silently dead for
        # every abbreviated name tested. Split on comma/slash first so a
        # dual-role "Dr. A, Dr. B" input matches either.
        doctor_values = _split_multi_value(doctor_name)
        if doctor_values:
            doctor_or = " OR ".join(["LOWER(dm.doctor_name) LIKE LOWER(%s)"] * len(doctor_values))
            score_terms.append(wrap(f"(CASE WHEN ({doctor_or}) THEN {10 * boost} ELSE 0 END)"))
            score_params.extend(f"%{dv}%" for dv in doctor_values)

    if department:
        # Stem-based LIKE, not equality: doctor_master.speciality is a full
        # descriptive phrase ("Traumatology, Joint Replacement & General
        # Orthopaedics"), never the app's standardized department name
        # ("Orthopaedics") -- confirmed exact equality scores ZERO for every
        # doctor checked regardless of what department value is passed,
        # silently discarding this signal entirely. Split on comma/slash
        # first so a dual-specialty "Cardiology, Nephrology" input can match
        # on either department instead of the whole compound string (which
        # never matches anything).
        dept_stems = [s for s in (_department_stem(dv) for dv in _split_multi_value(department)) if s]
        if dept_stems:
            dept_or = " OR ".join(["LOWER(dm.speciality) LIKE LOWER(%s)"] * len(dept_stems))
            score_terms.append(wrap(f"(CASE WHEN ({dept_or}) THEN {3 * boost} ELSE 0 END)"))
            score_params.extend(f"%{stem}%" for stem in dept_stems)

    if admission_type:
        # Dialysis Case / Day Case / General have fundamentally different cost
        # structures — matching within the same admission_type keeps the
        # comparison meaningful even when diagnosis text alone is thin.
        score_terms.append(wrap(f"(CASE WHEN LOWER(ia.admission_type) = LOWER(%s) THEN {4 * boost} ELSE 0 END)"))
        score_params.append(admission_type)

    if room_type:
        score_terms.append(
            wrap(
                """(CASE WHEN (
                    SELECT ba2.room_type FROM bed_allocations ba2
                    WHERE ba2.ip_id = ia.ip_id
                    ORDER BY ba2.start_date DESC, ba2.ba_id DESC LIMIT 1
                ) = %s THEN 1 ELSE 0 END)"""
            )
        )
        score_params.append(room_type)

    extra_score_sql = " + ".join(score_terms) if score_terms else "0"

    # Deliberately NOT bool(keywords): when every extracted keyword was common
    # (gating_keywords empty), no real keyword filter was ever applied, so
    # this must read as "no diagnosis signal" too, not just "no keywords".
    has_diagnosis_signal = bool(icd_codes) or bool(gating_keywords)

    logger.info(
        "find_similar_cases: match_path=%s icd_codes=%s keywords=%s gating_keywords=%s avg_signal_weight=%.2f "
        "weak_diagnosis_signal=%s doctor_id=%s doctor_name=%s department=%s room_type=%s admission_type=%s",
        "icd" if use_icd_path else ("keyword" if keywords else "no_diagnosis_signal"),
        icd_codes, keywords, gating_keywords, avg_signal_weight, weak_diagnosis_signal,
        doctor_id, doctor_name, department, room_type, admission_type,
    )

    if use_icd_path:
        cases = await _execute_icd_case_query(
            pool, weighted_codes, where_clause, where_params, extra_score_sql, score_params, limit,
        )
    else:
        cases = await _execute_case_query(pool, where_clause, where_params, extra_score_sql, score_params, limit)

    logger.info("find_similar_cases: primary query returned %d case(s)", len(cases))

    if cases:
        for case in cases:
            case["diagnosis_matched"] = has_diagnosis_signal
    elif has_diagnosis_signal and (doctor_id or doctor_name or department):
        logger.info(
            "find_similar_cases: primary match empty despite diagnosis signal -> "
            "falling back to doctor/department peer cases (no diagnosis requirement)"
        )
        # Diagnosis text matched nothing anywhere in discharge history (e.g. a
        # rare/oddly-worded diagnosis) — rather than leave the admission-time
        # estimate empty (Rs.0/confidence:none), fall back to the same doctor's
        # or department's peer cases with no diagnosis requirement, so there's
        # still a rough "near" bill to anchor on. These carry no diagnosis
        # signal, so callers must treat them as lower confidence than a
        # genuine keyword match (see diagnosis_matched flag below).
        fallback_conditions = ["ia.discharge_date IS NOT NULL", "ia.discharge_date != '0000-00-00'"]
        fallback_params = []
        peer_clauses = []
        if doctor_id:
            peer_clauses.append("dm.id = %s")
            fallback_params.append(doctor_id)
        elif doctor_name:
            # Substring match, not equality: callers send short names ("Dr. Mahin")
            # that are a prefix of the full doctor_master record ("Dr. Mahin
            # Nallasivam R R") — the same equality trap that starved the
            # primary department/room filters before they became ranking signals.
            # Split on comma/slash so a dual-role "Dr. A, Dr. B" input matches either.
            for dv in _split_multi_value(doctor_name):
                peer_clauses.append("LOWER(dm.doctor_name) LIKE LOWER(%s)")
                fallback_params.append(f"%{dv}%")
        if department:
            # department free text in doctor_master is inconsistent
            # ("Cardiologist" vs input "Cardiology", "Orthopedic Surgeon" vs
            # "Orthopaedics") — match on the first ~6 letters of the primary
            # word, which bridges common spelling/suffix variants without
            # requiring an exact label match. Split on comma/slash so a
            # dual-specialty "Cardiology, Nephrology" input matches on either.
            for dv in _split_multi_value(department):
                dept_stem = _department_stem(dv)
                if dept_stem:
                    peer_clauses.append("LOWER(dm.speciality) LIKE LOWER(%s)")
                    fallback_params.append(f"%{dept_stem}%")
        fallback_conditions.append(f"({' OR '.join(peer_clauses)})")
        if patient.get("ip_id"):
            fallback_conditions.append("ia.ip_id != %s")
            fallback_params.append(patient["ip_id"])
        fallback_where = " AND ".join(fallback_conditions)

        cases = await _execute_case_query(pool, fallback_where, fallback_params, extra_score_sql, score_params, limit)
        logger.info("find_similar_cases: fallback query returned %d case(s)", len(cases))
        for case in cases:
            case["diagnosis_matched"] = False

    # Fetch bill breakdowns for each similar case
    for case in cases:
        case["bills"] = await _fetch_case_bills(case["ip_id"])

    return cases


async def _execute_case_query(pool, where_clause: str, where_params: list, score_sql: str, score_params: list, limit: int) -> list[dict]:
    # ip_id is a final tiebreaker: match_score/discharge_date ties left the
    # LIMIT cutoff free to return a different subset of tied rows on repeat
    # calls with identical inputs, which fed a different anchor/prompt to
    # Gemini each time despite nothing about the patient actually changing.
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                f"""
                SELECT
                    ia.ip_id,
                    pp.patient_name,
                    pp.gender,
                    pp.age,
                    dm.speciality AS department,
                    dm.doctor_name,
                    (
                        SELECT ba.room_type FROM bed_allocations ba
                        WHERE ba.ip_id = ia.ip_id
                        ORDER BY ba.start_date DESC, ba.ba_id DESC LIMIT 1
                    ) AS room_type,
                    ia.payment_method AS payment_type,
                    ia.date AS admission_date,
                    ia.discharge_date,
                    ds.diagnosis,
                    ds.procedure_done,
                    DATEDIFF(ia.discharge_date, ia.date) as los_days,
                    ({score_sql}) AS match_score
                FROM ip_admissions ia
                JOIN discharge_summary ds ON ds.ip_id = ia.ip_id
                LEFT JOIN patient_profile pp ON pp.hr_no = ia.hr_no
                LEFT JOIN doctor_master dm
                    ON dm.id = COALESCE(NULLIF(ia.primary_doctor_id, 0), NULLIF(ia.doctor_id, 0))
                WHERE {where_clause}
                ORDER BY match_score DESC, ia.discharge_date DESC, ia.ip_id DESC
                LIMIT %s
                """,
                score_params + where_params + [limit],
            )
            return await cur.fetchall()


async def _execute_icd_case_query(
    pool, weighted_codes: list, where_clause: str, where_params: list,
    extra_score_sql: str, extra_score_params: list, limit: int,
) -> list[dict]:
    """ICD-code match scoring as a single aggregating JOIN instead of one
    correlated EXISTS subquery per code per candidate row — the latter
    measured 60-84s on real data (score computed for every WHERE-matching
    row before the DB could sort and LIMIT). Joining emr_diagnosis+
    diagnosis_master with icd_code IN (...) produces one row per
    (admission, matching code) pair; GROUP BY ip_id + SUM(weighted CASE)
    then computes the total match score in one pass.
    """
    codes = [code for code, _ in weighted_codes]
    case_whens = " ".join(f"WHEN %s THEN {weight}" for _, weight in weighted_codes)
    icd_case_sql = f"CASE dmi.icd_code {case_whens} ELSE 0 END"
    icd_placeholders = ", ".join(["%s"] * len(codes))

    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                f"""
                SELECT
                    ia.ip_id,
                    MAX(pp.patient_name) AS patient_name,
                    MAX(pp.gender) AS gender,
                    MAX(pp.age) AS age,
                    MAX(dm.speciality) AS department,
                    MAX(dm.doctor_name) AS doctor_name,
                    (
                        SELECT ba.room_type FROM bed_allocations ba
                        WHERE ba.ip_id = ia.ip_id
                        ORDER BY ba.start_date DESC, ba.ba_id DESC LIMIT 1
                    ) AS room_type,
                    MAX(ia.payment_method) AS payment_type,
                    MAX(ia.date) AS admission_date,
                    MAX(ia.discharge_date) AS discharge_date,
                    MAX(ds.diagnosis) AS diagnosis,
                    MAX(ds.procedure_done) AS procedure_done,
                    MAX(DATEDIFF(ia.discharge_date, ia.date)) AS los_days,
                    (SUM({icd_case_sql}) + {extra_score_sql}) AS match_score
                FROM ip_admissions ia
                JOIN discharge_summary ds ON ds.ip_id = ia.ip_id
                LEFT JOIN patient_profile pp ON pp.hr_no = ia.hr_no
                LEFT JOIN doctor_master dm
                    ON dm.id = COALESCE(NULLIF(ia.primary_doctor_id, 0), NULLIF(ia.doctor_id, 0))
                JOIN emr_diagnosis ed ON ed.hr_no = ia.hr_no
                    AND ed.created_date_time BETWEEN DATE_SUB(ia.date, INTERVAL 30 DAY) AND DATE_ADD(ia.discharge_date, INTERVAL 3 DAY)
                JOIN diagnosis_master dmi ON dmi.diagnosis_id = ed.diagnosis_id
                    AND dmi.icd_code IN ({icd_placeholders})
                WHERE {where_clause}
                GROUP BY ia.ip_id
                ORDER BY match_score DESC, MAX(ia.discharge_date) DESC, ia.ip_id DESC
                LIMIT %s
                """,
                codes + extra_score_params + codes + where_params + [limit],
            )
            return await cur.fetchall()


async def _fetch_case_bills(ip_id: int) -> dict:
    """Fetch total bill breakdown for a discharged case."""
    buckets = await _bill_buckets(ip_id)
    buckets["total"] = sum(buckets.values())
    return buckets


async def save_estimate(
    estimate_id: str,
    estimation_source: str,
    request_payload: dict,
    response_payload: dict,
    hr_no: str | None = None,
    request_signature: str | None = None,
    profile_signature: str | None = None,
    ip_id: int | None = None,
) -> None:
    """Persist a generated estimate in the destination database for audit purposes."""
    pool = await get_dest_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO estimates
                    (estimate_id, estimation_source, hr_no, request_payload, response_payload, created_at,
                     request_signature, profile_signature, ip_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    estimate_id,
                    estimation_source,
                    hr_no,
                    json.dumps(request_payload, default=str),
                    json.dumps(response_payload, default=str),
                    datetime.now(IST).replace(tzinfo=None),
                    request_signature,
                    profile_signature,
                    ip_id,
                ),
            )


async def fetch_recent_estimate_by_signature(request_signature: str, max_age_hours: float) -> dict | None:
    """Return the response_payload of the most recent estimate with the same
    request_signature, if it's within max_age_hours — used by estimate_manual
    to return the same result for an unchanged input instead of recomputing
    it (see request_signature comment in connection.py::_ensure_estimates_table).
    """
    pool = await get_dest_pool()
    cutoff = datetime.now(IST).replace(tzinfo=None) - timedelta(hours=max_age_hours)
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT TOP 1 response_payload FROM estimates
                WHERE request_signature = ? AND created_at >= ?
                ORDER BY created_at DESC
                """,
                (request_signature, cutoff),
            )
            row = await cur.fetchone()
            return json.loads(row[0]) if row else None


async def fetch_recent_estimate_by_ip_id(ip_id: int) -> dict | None:
    """Return the response_payload of the most recent estimate for this SPECIFIC
    admission (ip_id, not hr_no — a patient can be readmitted under the same
    hr_no, and ip_id is what actually disambiguates one admission from the
    next). No age cutoff: unlike the manual-estimate cache, this isn't a
    short-lived dedup — it's used by estimate_by_hr_no to keep repeated
    by-hr-no calls over the course of the SAME admission trending smoothly
    instead of swinging with each independent Gemini/peer-selection call
    (see HR_NO_RETRIGGER_BAND in estimator.py).
    """
    pool = await get_dest_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT TOP 1 response_payload FROM estimates
                WHERE ip_id = ?
                ORDER BY created_at DESC
                """,
                (ip_id,),
            )
            row = await cur.fetchone()
            return json.loads(row[0]) if row else None


async def fetch_recent_estimate_by_profile_signature(profile_signature: str, max_age_hours: float) -> dict | None:
    """Return {request_payload, response_payload} of the most recent estimate
    sharing profile_signature (same case, doctor excluded) within max_age_hours
    — used by estimate_manual's doctor-change guard (DOCTOR_CHANGE_BAND) to
    clamp a doctor-only change against the case's last quoted total.
    """
    pool = await get_dest_pool()
    cutoff = datetime.now(IST).replace(tzinfo=None) - timedelta(hours=max_age_hours)
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT TOP 1 request_payload, response_payload FROM estimates
                WHERE profile_signature = ? AND created_at >= ?
                ORDER BY created_at DESC
                """,
                (profile_signature, cutoff),
            )
            row = await cur.fetchone()
            if not row:
                return None
            return {"request_payload": json.loads(row[0]), "response_payload": json.loads(row[1])}


async def fetch_doctor_charges_by_disease(diagnosis_keyword: str = "") -> list[dict]:
    """Fetch doctor charges analysis grouped by disease.

    Returns: list of dicts with:
        - diagnosis: primary diagnosis
        - doctor_name: treating doctor
        - los_days: length of stay
        - doctor_total: total doctor charge
        - doctor_per_day: doctor_total / los_days
        - patient_age: age of patient
        - admission_date: when admitted
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            where_clause = ""
            params = []

            if diagnosis_keyword:
                where_clause = "AND ds.diagnosis LIKE %s"
                params = [f"%{diagnosis_keyword}%"]

            await cur.execute(
                f"""
                SELECT
                    ds.diagnosis,
                    dm.doctor_name,
                    DATEDIFF(ia.discharge_date, ia.date) AS los_days,
                    COALESCE(SUM(CASE
                        WHEN LOWER(TRIM(bpm.category_name)) IN ('consultation', 'doctors surgical fees') THEN a.credit
                        WHEN a.bp_id IS NULL THEN 0
                        ELSE 0
                    END), 0) AS doctor_total,
                    pp.age AS patient_age,
                    ia.date AS admission_date,
                    ia.discharge_date,
                    pp.patient_name
                FROM ip_admissions ia
                JOIN discharge_summary ds ON ds.ip_id = ia.ip_id
                LEFT JOIN patient_profile pp ON pp.hr_no = ia.hr_no
                LEFT JOIN doctor_master dm ON dm.id = COALESCE(NULLIF(ia.primary_doctor_id, 0), NULLIF(ia.doctor_id, 0))
                LEFT JOIN accounts a ON a.ip_id = ia.ip_id AND a.credit > 0
                LEFT JOIN bill_particular_master bpm ON bpm.bp_id = a.bp_id
                WHERE ia.discharge_date IS NOT NULL
                    AND ia.discharge_date != '0000-00-00'
                    AND DATEDIFF(ia.discharge_date, ia.date) > 0
                    {where_clause}
                GROUP BY ia.ip_id, ds.diagnosis, dm.doctor_name, ia.date, ia.discharge_date, pp.age, pp.patient_name
                ORDER BY ia.discharge_date DESC
                LIMIT 50
                """,
                params,
            )
            rows = await cur.fetchall()

    result = []
    for row in rows:
        los_days = row.get("los_days") or 1
        doctor_total = float(row.get("doctor_total") or 0)
        result.append({
            "diagnosis": row.get("diagnosis"),
            "doctor_name": row.get("doctor_name"),
            "patient_name": row.get("patient_name"),
            "los_days": los_days,
            "doctor_total": doctor_total,
            "doctor_per_day": doctor_total / los_days if los_days > 0 else 0,
            "patient_age": row.get("patient_age"),
            "admission_date": str(row.get("admission_date")) if row.get("admission_date") else None,
            "discharge_date": str(row.get("discharge_date")) if row.get("discharge_date") else None,
        })

    return result

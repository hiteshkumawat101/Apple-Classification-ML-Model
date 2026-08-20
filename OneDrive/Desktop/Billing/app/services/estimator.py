import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from statistics import median

from app.db.queries import (
    fetch_patient_by_hr_no,
    fetch_current_bills,
    fetch_admission_utilization,
    fetch_utilization_intensity,
    find_similar_cases,
    fetch_room_rate,
    fetch_doctor_round_rate,
    save_estimate,
    fetch_recent_estimate_by_signature,
    fetch_recent_estimate_by_profile_signature,
    fetch_recent_estimate_by_ip_id,
    fetch_diagnosis_component_support,
    fetch_doctor_charges_by_disease,
)
from app.services.prompt_builder import build_estimation_prompt
from app.services.gemini_client import get_estimation
from app.services.component_rules import COMPONENT_RULES, EVIDENCE_KEYWORDS
from app.models.response import EstimateResponse, BillComponent, ComponentBills
from app.models.request import InitialEstimateRequest


logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

BILL_KEYS = ["room", "doctor", "laboratory", "radiology", "operating_theatre", "dialysis", "physiotherapy", "miscellaneous"]

# find_similar_cases's match_score ties heavily within a diagnosis (doctor/
# department/room scoring dominates once a diagnosis keyword matches at all),
# so a plain "top 10 by match_score" is effectively an arbitrary subset of a
# much larger tied group, broken only by discharge_date recency -- confirmed
# on real Acute Ischemic Stroke data: all 10 top-ranked peers shared the
# identical match_score, and that recency-tiebroken subset skewed toward
# longer/costlier stays than the broader tied pool (anchor landed at 2.4x the
# true 25-case median vs 1.3x once the anchor was computed over 25 peers
# instead of 10). Fetching a WIDER pool for the anchor/confidence/intensity
# calculations (cheap, pure-Python aggregation) while still only showing
# Gemini the top SIMILAR_CASES_PROMPT_LIMIT in the prompt keeps prompt token
# cost/latency unchanged.
SIMILAR_CASES_ANCHOR_LIMIT = 25
SIMILAR_CASES_PROMPT_LIMIT = 10


ANCHOR_BAND = 0.20

# LOS-override requests need a much tighter band than a normal estimate: the
# anchor here is per_day_rate * target_los, strictly monotonic in target_los,
# but a +/-20% band leaves enough overlap between e.g. an 18-day and a 15-day
# request (anchor15 * 1.20 > anchor18 * 0.80 whenever the LOS ratio exceeds
# ~0.83) that two independent Gemini samples can land in the "wrong" relative
# order — observed directly: reducing 18 days to 15 produced a HIGHER bill.
# +/-8% keeps that overlap window to LOS deltas of a day or two, which is an
# acceptable residual risk clinicians would treat as noise anyway.
LOS_OVERRIDE_ANCHOR_BAND = 0.08
LOS_OVERRIDE_FIXED_COMPONENT_BAND = 0.02


INTENSITY_RATIO_BOUNDS = (0.5, 2.0)


# estimate_manual has no admission record to anchor to (unlike estimate_by_hr_no,
# where current_bills grounds every call). Without this, toggling a field back
# to its original value and re-submitting was observed to return a DIFFERENT
# number than the original call — driven by peer-selection LIMIT ties and by
# Gemini not being perfectly reproducible even at temperature=0 (see
# gemini_client.py). Caching by a signature of the clinically material fields
# guarantees the same input returns the same output instead of relying on
# those two layers to individually stay stable.
MANUAL_ESTIMATE_CACHE_TTL_HOURS = 24


# When only the doctor changes for an otherwise identical case (same
# diagnosis/department/room/payment/age/gender/admission_type), the new
# doctor's peer set can price the case very differently from the one already
# quoted, even though nothing clinically changed. Clamping the new total to
# within this band of the LAST quote for the same case (doctor aside) keeps a
# doctor swap a refinement rather than a re-roll. Pick 0.10 instead if a
# tighter cap is wanted.
DOCTOR_CHANGE_BAND = 0.15


# estimate_by_hr_no is re-called repeatedly over the course of ONE admission
# (each time a clinician checks in on the patient's projected bill) — every
# call independently re-derives the anchor from peer cases and re-samples
# Gemini, so the same underlying admission could otherwise get a noticeably
# different final-bill projection today vs. yesterday even though nothing
# material changed, just as the doctor-change and LOS-override cases above
# showed independent re-derivation drifting. Clamping today's prediction to
# within this band of the last quote for the SAME ip_id (see
# fetch_recent_estimate_by_ip_id) keeps day-to-day retriggers a refinement
# that tracks the growing current_bills/utilization rather than a re-roll.
HR_NO_RETRIGGER_BAND = 0.15


def resolve_component_eligibility(
    diagnosis: str,
    utilization: dict | None,
    component_support: dict | None,
) -> dict[str, dict]:
    """Determine eligibility (mandatory/conditional_met/conditional_unmet/not_applicable/unknown)
    for each component based on the diagnosis, utilization, and historical support data.

    Returns: {bucket: {"status": str, "eligible": bool}}
    - status: one of "mandatory", "conditional_met", "conditional_unmet", "not_applicable", "unknown"
    - eligible: True if the component should be charged, False otherwise

    Does not modify prediction or validation state; purely a decision function.
    """
    eligibility = {}
    diagnosis_lower = (diagnosis or "").lower()

    for bucket, rule in COMPONENT_RULES.items():
        rule_type = rule.get("type")

        if rule_type == "mandatory":
            eligibility[bucket] = {"status": "mandatory", "eligible": True}

        elif rule_type == "conditional":
            evidence_keywords = rule.get("evidence", [])
            has_evidence = False

            # Check diagnosis text for evidence
            for evidence_name in evidence_keywords:
                keywords = EVIDENCE_KEYWORDS.get(evidence_name, [])
                for keyword in keywords:
                    if keyword.lower() in diagnosis_lower:
                        has_evidence = True
                        break
                if has_evidence:
                    break

            # Check utilization counters if available
            if not has_evidence and utilization:
                for evidence_name in evidence_keywords:
                    if evidence_name == "ot_procedures_total":
                        if utilization.get("ot_procedures_total", 0) > 0:
                            has_evidence = True
                            break
                    elif evidence_name == "dialysis_sessions":
                        if utilization.get("dialysis_sessions", 0) > 0:
                            has_evidence = True
                            break

            status = "conditional_met" if has_evidence else "conditional_unmet"
            eligibility[bucket] = {"status": status, "eligible": has_evidence}

        elif rule_type == "diagnosis_based":
            if component_support is None:
                status = "unknown"
                eligible = True  # preserve "don't suppress on unknown" behavior
            else:
                # component_support[bucket] is True (supported) or False (not supported)
                is_supported = component_support.get(bucket, False)
                if is_supported:
                    status = "supported_historically"
                    eligible = True
                else:
                    status = "not_applicable"
                    eligible = False

            eligibility[bucket] = {"status": status, "eligible": eligible}

        else:
            eligibility[bucket] = {"status": "unknown", "eligible": True}

    return eligibility


def enforce_component_eligibility(
    prediction: dict,
    eligibility: dict,
    context: str,
) -> dict:
    """Zero out components marked ineligible and fold their amounts into miscellaneous.

    Preserves the total_bill/hospital_bill invariant: the amount is reclassified,
    not deleted. Records zeroed components in prediction["_zeroed_components"] for
    later reference (used in _build_response to prevent re-pricing).

    Returns the modified prediction dict.
    """
    if not eligibility:
        return prediction

    comp = prediction.get("component_bills", {})
    new_comp = dict(comp)
    zeroed_components = set(prediction.get("_zeroed_components", set()))
    reclassified = 0.0

    for bucket in BILL_KEYS:
        bucket_eligible = eligibility.get(bucket, {}).get("eligible", True)
        amount = float(comp.get(bucket, 0))

        if not bucket_eligible and amount > 0:
            status = eligibility.get(bucket, {}).get("status", "unknown")
            logger.warning(
                "[%s] enforce_component_eligibility: '%s'=%.2f marked ineligible (status=%s) "
                "-> moving to miscellaneous",
                context, bucket, amount, status,
            )
            reclassified += amount
            new_comp[bucket] = 0.0
            zeroed_components.add(bucket)

    if reclassified <= 0:
        return prediction

    new_comp["miscellaneous"] = float(new_comp.get("miscellaneous", 0)) + reclassified

    prediction = dict(prediction)
    prediction["component_bills"] = new_comp
    prediction["_zeroed_components"] = zeroed_components
    logger.info(
        "[%s] enforce_component_eligibility: reclassified Rs.%.2f to miscellaneous, "
        "zeroed_components=%s", context, reclassified, zeroed_components,
    )
    return prediction


async def _intensity_ratio(current_ip_id: int, elapsed_days: float, similar_cases: list[dict]) -> float:
    """How much busier/quieter this patient is than their diagnosis-matched
    peers, using utilization counts (investigations/medications/OT/dialysis)
    rather than diagnosis text — a cheap complexity signal that doesn't
    depend on the diagnosis field actually naming a primary condition.
    Bounded to avoid a noisy small-sample ratio swinging the anchor wildly.
    """
    current_intensity = await fetch_utilization_intensity(current_ip_id)
    current_per_day = current_intensity / max(elapsed_days, 1)

    weighted_sum = 0.0
    weight_total = 0.0
    for case in similar_cases:
        los = case.get("los_days") or 0
        if not los or los <= 0:
            continue
        weight = float(case.get("match_score") or 0) + 1
        peer_intensity = await fetch_utilization_intensity(case["ip_id"])
        weighted_sum += weight * (peer_intensity / los)
        weight_total += weight

    if weight_total == 0 or weighted_sum == 0:
        logger.info(
            "intensity_ratio[ip_id=%s]: no usable peer utilization data (weight_total=%.2f, "
            "weighted_sum=%.2f) -> defaulting to 1.0", current_ip_id, weight_total, weighted_sum,
        )
        return 1.0

    peer_per_day = weighted_sum / weight_total
    if peer_per_day <= 0:
        logger.info(
            "intensity_ratio[ip_id=%s]: peer_per_day=%.4f <= 0 -> defaulting to 1.0",
            current_ip_id, peer_per_day,
        )
        return 1.0

    ratio = current_per_day / peer_per_day
    bounded_ratio = min(max(ratio, INTENSITY_RATIO_BOUNDS[0]), INTENSITY_RATIO_BOUNDS[1])
    logger.info(
        "intensity_ratio[ip_id=%s]: current_intensity=%.2f over %.1f elapsed days (%.4f/day), "
        "peer_per_day=%.4f (from %d peer cases) -> raw_ratio=%.4f, bounded_ratio=%.4f",
        current_ip_id, current_intensity, elapsed_days, current_per_day, peer_per_day,
        len(similar_cases), ratio, bounded_ratio,
    )
    return bounded_ratio


async def _calculate_doctor_per_day_from_history(similar_cases: list[dict]) -> float | None:
	"""Calculate actual doctor fee per day from historical similar cases.

	Uses real billing data to validate/adjust the master rate. Returns the
	weighted median per-day doctor charge across similar cases.
	"""
	doctor_rates = []
	for case in similar_cases:
		los = case.get("los_days") or 0
		if not los or los <= 0:
			continue
		doctor_charge = case.get("bills", {}).get("doctor", 0) or 0
		if doctor_charge > 0:
			weight = float(case.get("match_score") or 0) + 1
			per_day = doctor_charge / los
			doctor_rates.append((per_day, weight))

	if not doctor_rates:
		logger.info("doctor_per_day_from_history: no similar cases with doctor charges -> using system rate")
		return None

	per_day_rate = _weighted_median(doctor_rates)
	logger.info(
		"doctor_per_day_from_history: calculated from %d cases -> %.2f/day",
		len(doctor_rates), per_day_rate,
	)
	return per_day_rate


def _weighted_median(values_and_weights: list[tuple[float, float]]) -> float:
    """Weighted median: the value at the point where cumulative weight first
    reaches half the total weight. Unlike a weighted mean, a handful of
    extreme values can't drag it arbitrarily far from the typical case --
    at most it moves to the next-ranked value, however extreme the outliers
    themselves are.
    """
    sorted_vw = sorted(values_and_weights, key=lambda vw: vw[0])
    total_weight = sum(w for _, w in sorted_vw)
    if total_weight <= 0:
        return 0.0
    half = total_weight / 2
    cumulative = 0.0
    for value, weight in sorted_vw:
        cumulative += weight
        if cumulative >= half:
            return value
    return sorted_vw[-1][0]


def _similarity_anchor(
    similar_cases: list[dict],
    current_total: float = 0.0,
    elapsed_days: float = 0.0,
    target_los: float | None = None,
    intensity_ratio: float = 1.0,
) -> float | None:
    """Similarity-weighted anchor = bill already incurred + a peer-informed
    projection of what's left.

    A flat average of similar cases' totals was tested and found to badly
    mislead short admissions toward the average of longer peers (e.g. a
    2-day stay pulled toward a pool that includes 8-day stays) — that's a
    duration mismatch, not a similarity failure. Normalizing to a per-day
    rate from peers, then projecting only the REMAINING days onto the
    patient's OWN bill-to-date, backtested far closer to actual outcomes.

    target_los overrides the peer-average LOS used for that projection —
    pass the clinician's overridden LOS here so the anchor (and therefore
    the +/-20% clamp band) shifts down for a shorter stay and up for a
    longer one, instead of staying pinned to the peer average regardless
    of which direction the override moves.

    intensity_ratio scales the per-day rate up/down for a patient busier or
    quieter than their peers (see _intensity_ratio) — a partial mitigation
    for diagnosis text that carries no cost signal (e.g. a bare comorbidity
    list with no primary condition named).

    A 1-day stay with a big one-time cost front-loaded on that single day
    (e.g. an OT procedure) produces a per-day rate of the ENTIRE bill divided
    by 1 — backtested stroke cases showed this inflating the rate to
    Rs.155,802/day against a Rs.15-47k/day norm for multi-day peers. Flooring
    the RATE denominator (not the LOS estimate itself) keeps a genuine 1-day
    admission's cost from being read as its true daily rate.

    The per-day rate itself uses a WEIGHTED MEDIAN, not a weighted mean: for
    a right-skewed diagnosis (a handful of very long, very expensive peers
    alongside many ordinary ones -- confirmed on real Acute Ischemic Stroke
    and Severe Pneumonia data, where the mean-based anchor landed the final
    estimate at 2.2-2.7x the broader cohort's true median), a mean lets those
    few outliers pull the anchor arbitrarily far from what a TYPICAL peer
    actually cost, and match_score barely discriminates within a diagnosis
    (most peers score near-identically) so the spike gets full weight rather
    than being diluted. The median represents the typical peer regardless of
    how extreme the outliers in the tail are.
    """
    MIN_LOS_FOR_RATE = 2.0

    rates_and_weights = []
    los_and_weights = []
    weight_total = 0.0
    for case in similar_cases:
        los = case.get("los_days") or 0
        if not los or los <= 0:
            continue
        weight = float(case.get("match_score") or 0) + 1  # +1: a 0-score case still counts a little
        total = case.get("bills", {}).get("total", 0)
        rate = total / max(los, MIN_LOS_FOR_RATE)
        rates_and_weights.append((rate, weight))
        los_and_weights.append((los, weight))
        weight_total += weight

    if weight_total == 0:
        logger.info("similarity_anchor: no similar case had usable los_days -> anchor=None")
        return None

    per_day_rate = _weighted_median(rates_and_weights) * intensity_ratio
    # expected_los also uses the median for the same reason as the rate above:
    # a handful of unusually long stays (e.g. 18-21 days) pull a weighted MEAN
    # los upward same as they pull the rate, compounding the same skew twice.
    expected_los = target_los if target_los is not None else _weighted_median(los_and_weights)
    remaining_days = max(expected_los - elapsed_days, 0)
    anchor = current_total + per_day_rate * remaining_days
    logger.info(
        "similarity_anchor: weight_total=%.2f, per_day_rate=%.2f (intensity_ratio=%.4f), "
        "expected_los=%.2f (target_los=%s), elapsed_days=%.2f, remaining_days=%.2f, "
        "current_total=%.2f -> anchor=%.2f",
        weight_total, per_day_rate, intensity_ratio, expected_los, target_los, elapsed_days,
        remaining_days, current_total, anchor,
    )
    return anchor


def _override_anchor(old_los: float, old_estimate: float, new_los_days: float) -> float | None:
    """Anchors a LOS-override prediction to the clinician's OWN prior estimate
    instead of a peer-derived anchor recomputed from scratch on every call.

    A peer-based anchor (see _similarity_anchor) was observed to swing between
    independent calls — even reproducing a DIFFERENT number for the exact same
    new_los_days requested twice — and could rank a shorter LOS higher than a
    longer one despite the tightened override band. Scaling old_estimate's
    per-day rate linearly by new_los_days is monotonic by construction and
    reproduces old_estimate exactly when new_los_days == old_los.
    """
    if not old_los or old_los <= 0:
        logger.info("override_anchor: old_los=%s not usable -> anchor=None", old_los)
        return None

    per_day_rate = old_estimate / old_los
    anchor = per_day_rate * new_los_days
    logger.info(
        "override_anchor: old_los=%.2f old_estimate=%.2f per_day_rate=%.2f new_los_days=%.2f -> anchor=%.2f",
        old_los, old_estimate, per_day_rate, new_los_days, anchor,
    )
    return anchor


def _clamp_to_anchor(prediction: dict, anchor: float | None, band: float = ANCHOR_BAND) -> dict:
    """Enforce the +/-band anchor range in code rather than relying on the LLM
    to obey a numeric instruction in the prompt text — verified by backtest
    that Gemini does NOT reliably respect a stated "must stay within X%"
    constraint (observed a prediction 3.5x the anchor despite the instruction).
    Rescales the whole breakdown proportionally so components still sum
    correctly; Gemini still determines the distribution across components,
    only the overall scale is corrected.
    """
    if not anchor or anchor <= 0:
        logger.info("clamp: no usable anchor (anchor=%s) -> Gemini prediction passed through unclamped", anchor)
        return prediction

    total = float(prediction.get("total_bill", 0))
    if total <= 0:
        logger.warning("clamp: Gemini total_bill=%.2f is non-positive against anchor=%.2f -> passed through unclamped", total, anchor)
        return prediction

    low, high = anchor * (1 - band), anchor * (1 + band)
    if low <= total <= high:
        logger.info(
            "clamp: Gemini total_bill=%.2f within band [%.2f, %.2f] (anchor=%.2f, band=+/-%.0f%%) -> no adjustment",
            total, low, high, anchor, band * 100,
        )
        return prediction

    clamped_total = min(max(total, low), high)
    scale = clamped_total / total
    logger.warning(
        "clamp: Gemini total_bill=%.2f OUTSIDE band [%.2f, %.2f] (anchor=%.2f, band=+/-%.0f%%) "
        "-> rescaled to %.2f (scale=%.4f)",
        total, low, high, anchor, band * 100, clamped_total, scale,
    )

    prediction = dict(prediction)
    prediction["total_bill"] = clamped_total
    prediction["hospital_bill"] = float(prediction.get("hospital_bill", 0)) * scale
    prediction["pharmacy_bill"] = float(prediction.get("pharmacy_bill", 0)) * scale
    prediction["component_bills"] = {
        k: float(v) * scale for k, v in prediction.get("component_bills", {}).items()
    }
    return prediction


def _clamp_los_to_anchor(los_days: float, anchor: float | None, band: float) -> float:
    """Same guard as _clamp_to_anchor but for a bare scalar (expected_total_los_days)
    rather than a bill breakdown — total_bill landing inside its band doesn't
    keep Gemini's independently-predicted LOS in check (observed the same
    doctor-change case swing from an 8-day to a 16-day stay while the total
    bill stayed close, since bill and LOS are predicted independently).
    """
    if not anchor or anchor <= 0:
        return los_days
    low, high = anchor * (1 - band), anchor * (1 + band)
    return min(max(los_days, low), high)


async def _clamp_los_against_historical(
    prediction: dict,
    similar_cases: list[dict] | None,
    diagnosis: str,
    context: str,
) -> dict:
    """Sanity check: if Gemini's predicted LOS exceeds the P90 of historical similar cases,
    clamp it to P75 to avoid over-predictions on complex cases like Stroke or Pneumonia.

    This prevents Gemini from predicting 25-30 day stays when historical average is 8-12 days,
    which inflates room/doctor charges and total bill by 126-138%.
    """
    if not similar_cases or not prediction:
        return prediction

    gemini_los = float(prediction.get("expected_total_los_days", 0))
    if gemini_los <= 0:
        return prediction

    # Extract historical LOS from similar cases
    historical_los = sorted([c.get("los_days") for c in similar_cases if c.get("los_days") and c.get("los_days") > 0])

    if len(historical_los) < 5:
        logger.info("[%s] LOS sanity check skipped: insufficient historical data (%d cases)", context, len(historical_los))
        return prediction

    # Calculate P75 and P90
    from statistics import quantiles
    percentiles = quantiles(historical_los, n=4)
    p75_los = percentiles[2]  # 75th percentile
    p90_idx = int(0.9 * len(historical_los))
    p90_los = historical_los[p90_idx] if p90_idx < len(historical_los) else historical_los[-1]

    # Check if Gemini LOS exceeds P90 (outlier territory)
    if gemini_los > p90_los:
        logger.warning(
            "[%s] LOS sanity check triggered: Gemini predicted %.1f days, "
            "but P90 for '%s' is %.1f days (P75=%.1f). Clamping to P75.",
            context, gemini_los, diagnosis, p90_los, p75_los,
        )
        prediction = dict(prediction)
        prediction["expected_total_los_days"] = p75_los

        # Recalculate affected components
        # (room, doctor, physiotherapy are LOS-dependent)
        logger.info(
            "[%s] LOS clamped: %.1f days → %.1f days (P90=%.1f). "
            "Room/doctor/physiotherapy charges will be recalculated.",
            context, gemini_los, p75_los, p90_los,
        )

    return prediction


def _confidence_band(num_cases: int, diagnosis_matched: bool = True) -> str:
    if num_cases == 0:
        return "none"
    if not diagnosis_matched:
        # find_similar_cases() fell back to doctor/department peers with no
        # diagnosis/procedure keyword overlap at all — a rough "near" bill,
        # never "high" confidence regardless of how many peers were found.
        return "limited"
    if num_cases >= 10:
        return "high"
    elif num_cases >= 5:
        return "moderate"
    return "limited"


def _empty_response(estimate_id: str, estimation_source: str) -> EstimateResponse:
    return EstimateResponse(
        estimate_id=estimate_id,
        estimation_source=estimation_source,
        as_of=datetime.now(IST).isoformat(),
        confidence_band="none",
        confidence_score=0,
        total_bill=BillComponent(),
        hospital_bill=BillComponent(),
        pharmacy_bill=BillComponent(),
        component_bills=ComponentBills(),
    )


def _build_response(
    estimate_id: str,
    estimation_source: str,
    band: str,
    current_bills: dict,
    prediction: dict,
    los_days_elapsed: float = 0.0,
    los_override_days: float | None = None,
    room_per_day: float | None = None,
    doctor_per_day: float | None = None,
    similar_cases: list[dict] | None = None,
    component_support: dict[str, bool] | None = None,
    eligibility: dict[str, dict] | None = None,
) -> EstimateResponse:
    comp = dict(prediction.get("component_bills", {}))

    if los_override_days is not None:
        # The clinician's own override IS the total LOS, even if it's below
        # elapsed_days — that case correctly floors expected_remaining_los to
        # 0 below, it's not a contradiction to flag.
        expected_total_los = los_override_days
    else:
        # Gemini predicts expected_total_los_days from peer LOS patterns with
        # no hard link to this admission's OWN elapsed_days — observed on a
        # real 12-day-elapsed ICU admission returning expected_total_los_days=8,
        # a self-contradictory "expected to stay less time than already
        # spent" figure the API would otherwise surface as-is. Floor it to
        # what's already elapsed since the patient is still admitted.
        raw_expected_total_los = float(prediction.get("expected_total_los_days", 0))
        expected_total_los = max(raw_expected_total_los, los_days_elapsed)
        if expected_total_los != raw_expected_total_los:
            logger.warning(
                "expected_total_los_days=%.2f from Gemini is below los_days_elapsed=%.2f "
                "(patient still admitted) -> floored to %.2f",
                raw_expected_total_los, los_days_elapsed, expected_total_los,
            )
    expected_remaining_los = max(expected_total_los - los_days_elapsed, 0)
    current_hospital = sum(float(current_bills.get(k, 0)) for k in BILL_KEYS)
    current_pharmacy = float(current_bills.get("pharmacy", 0))
    current_total = current_hospital + current_pharmacy

    hospital_final = float(prediction.get("hospital_bill", 0))
    pharmacy_final = float(prediction.get("pharmacy_bill", 0))
    total_final = float(prediction.get("total_bill", 0))

    def _clamp_scalar_band(value: float, anchor: float, band: float) -> float:
        if anchor <= 0:
            return value
        low, high = anchor * (1 - band), anchor * (1 + band)
        return min(max(value, low), high)

    if similar_cases and expected_total_los > 0:
        valid_cases = [c for c in similar_cases if c.get("los_days") and c["los_days"] > 0]

        if valid_cases:
            baseline_los = median([c["los_days"] for c in valid_cases])
            los_delta = expected_total_los - baseline_los

            def bucket_avg(key: str) -> float:
                vals = [float(c.get("bills", {}).get(key, 0)) for c in valid_cases]
                return sum(vals) / len(vals) if vals else 0.0

            def bucket_per_day_avg(key: str) -> float:
                vals = [
                    float(c.get("bills", {}).get(key, 0)) / float(c["los_days"])
                    for c in valid_cases
                    if c.get("los_days")
                ]
                return sum(vals) / len(vals) if vals else 0.0

            system_comp = dict(comp)
            predicted_comp = {
                "room": float(comp.get("room", 0)),
                "doctor": float(comp.get("doctor", 0)),
                "laboratory": float(comp.get("laboratory", 0)),
                "radiology": float(comp.get("radiology", 0)),
                "operating_theatre": float(comp.get("operating_theatre", 0)),
                "dialysis": float(comp.get("dialysis", 0)),
                "physiotherapy": float(comp.get("physiotherapy", 0)),
                "miscellaneous": float(comp.get("miscellaneous", 0)),
            }
            predicted_pharmacy = pharmacy_final

            if room_per_day is not None and room_per_day > 0:
                system_comp["room"] = room_per_day * expected_total_los

            doctor_baseline = bucket_avg("doctor")
            # Fallback to system rate if peer data missing
            if doctor_baseline <= 0 and doctor_per_day and doctor_per_day > 0:
                doctor_baseline = doctor_per_day * expected_total_los

            doctor_variable_rate = (
                DOCTOR_OVERRIDE_PERCENT * doctor_per_day
                if doctor_per_day is not None and doctor_per_day > 0
                else DOCTOR_OVERRIDE_PERCENT * bucket_per_day_avg("doctor")
            )

            system_comp["doctor"] = max(doctor_baseline + doctor_variable_rate * los_delta, 0.0)
            lab_baseline = bucket_avg("laboratory")
            lab_adjusted = lab_baseline + LAB_OVERRIDE_PERCENT * bucket_per_day_avg("laboratory") * los_delta
            # Cap LOS-driven adjustments at ±40% to prevent extreme values
            lab_capped = max(min(lab_adjusted, lab_baseline * 1.4), max(lab_baseline * 0.6, 0.0))
            if lab_adjusted != lab_capped:
                logger.info(
                    "[%s] laboratory adjustment capped: baseline=%.2f adjusted=%.2f capped=%.2f",
                    ctx, lab_baseline, lab_adjusted, lab_capped,
                )
            system_comp["laboratory"] = lab_capped
            # For fixed components (radiology, OT, dialysis): use historical average if eligible,
            # respecting the eligibility decision from earlier pipeline stages.
            # This prevents under-prediction when Gemini misses that a diagnosis needs the component,
            # while honoring explicit zero decisions from validation stages.
            zeroed_components = prediction.get("_zeroed_components", set())
            is_radiology_eligible = eligibility.get("radiology", {}).get("eligible", True) if eligibility else (not component_support or component_support.get("radiology", True))
            if is_radiology_eligible and "radiology" not in zeroed_components:
                radiology_baseline = bucket_avg("radiology")
                # If eligible but similar cases show no radiology charges, use a minimum default
                # This handles diagnoses where radiology IS needed but peers simply don't have it recorded
                if radiology_baseline <= 0:
                    radiology_baseline = 2500.0  # Default minimum radiology charge
                system_comp["radiology"] = max(radiology_baseline, 0.0)
            else:
                system_comp["radiology"] = 0.0

            is_ot_eligible = eligibility.get("operating_theatre", {}).get("eligible", True) if eligibility else (not component_support or component_support.get("operating_theatre", True))
            if is_ot_eligible and "operating_theatre" not in zeroed_components:
                system_comp["operating_theatre"] = max(bucket_avg("operating_theatre"), 0.0)
            else:
                system_comp["operating_theatre"] = 0.0

            is_dialysis_eligible = eligibility.get("dialysis", {}).get("eligible", True) if eligibility else (not component_support or component_support.get("dialysis", True))
            if is_dialysis_eligible and "dialysis" not in zeroed_components:
                system_comp["dialysis"] = max(bucket_avg("dialysis"), 0.0)
            else:
                system_comp["dialysis"] = 0.0

            # For variable components, apply LOS adjustment if eligible for this diagnosis.
            # This prevents under-prediction when Gemini misses that a diagnosis needs the component.
            is_physio_eligible = eligibility.get("physiotherapy", {}).get("eligible", True) if eligibility else (not component_support or component_support.get("physiotherapy", True))
            if is_physio_eligible and "physiotherapy" not in zeroed_components:
                physio_baseline = bucket_avg("physiotherapy")
                # If eligible but similar cases show no physiotherapy charges, use a minimum default
                # This handles diagnoses where physiotherapy IS needed but peers simply don't have it recorded
                if physio_baseline <= 0:
                    physio_baseline = 1500.0  # Default minimum physiotherapy charge per case
                physio_per_day = bucket_per_day_avg("physiotherapy")
                if physio_per_day <= 0:
                    physio_per_day = 500.0  # Default per-day physiotherapy rate if not in peers
                physio_adjusted = physio_baseline + physio_per_day * los_delta
                # Cap LOS-driven adjustments at ±40% to prevent extreme values
                physio_capped = max(min(physio_adjusted, physio_baseline * 1.4 if physio_baseline > 0 else physio_adjusted),
                                    max(physio_baseline * 0.6 if physio_baseline > 0 else 0, 0.0))
                if physio_adjusted != physio_capped:
                    logger.info(
                        "[%s] physiotherapy adjustment capped: baseline=%.2f adjusted=%.2f capped=%.2f",
                        ctx, physio_baseline, physio_adjusted, physio_capped,
                    )
                system_comp["physiotherapy"] = physio_capped
            else:
                system_comp["physiotherapy"] = 0.0
            system_comp["miscellaneous"] = max(
                bucket_avg("miscellaneous") + bucket_per_day_avg("miscellaneous") * los_delta,
                0.0,
            )
            pharmacy_final = max(
                bucket_avg("pharmacy") + PHARMACY_OVERRIDE_PERCENT * bucket_per_day_avg("pharmacy") * los_delta,
                0.0,
            )

            if los_override_days is not None:
                fixed_component_anchors = {
                    "doctor": max(bucket_avg("doctor"), 0.0),
                    "radiology": max(bucket_avg("radiology"), 0.0),
                    "operating_theatre": max(bucket_avg("operating_theatre"), 0.0),
                    "dialysis": max(bucket_avg("dialysis"), 0.0),
                }
                for key, anchor in fixed_component_anchors.items():
                    original = float(system_comp.get(key, 0))
                    clamped = _clamp_scalar_band(
                        original, anchor, LOS_OVERRIDE_FIXED_COMPONENT_BAND
                    )
                    system_comp[key] = clamped
                    if clamped != original:
                        logger.info(
                            "los_override_fixed_component_guard: component=%s value=%.2f "
                            "outside +/-%.0f%% of anchor=%.2f -> clamped to %.2f",
                            key, original, LOS_OVERRIDE_FIXED_COMPONENT_BAND * 100,
                            anchor, clamped,
                        )

            for key in BILL_KEYS:
                system_comp[key] = max(float(system_comp.get(key, 0)), float(current_bills.get(key, 0)))

            pharmacy_final = max(pharmacy_final, current_pharmacy)
            comp = system_comp
            hospital_final = sum(float(comp.get(k, 0)) for k in BILL_KEYS)
            total_final = hospital_final + pharmacy_final

            # Sanity check: component bills should sum to hospital bill
            component_sum = sum(float(comp.get(k, 0)) for k in BILL_KEYS)
            if abs(component_sum - hospital_final) > 0.01:  # Allow 1 paisa rounding error
                logger.warning(
                    "[%s] component sum (%.2f) does not match hospital_bill (%.2f) - possible data inconsistency",
                    ctx, component_sum, hospital_final,
                )

            logger.info(
                "system_los_component_pricing: baseline_los=%.2f expected_total_los_days=%.2f "
                "los_delta=%.2f room_per_day=%.2f doctor_per_day=%.2f "
                "predicted_components=%s predicted_pharmacy=%.2f -> overridden_components=%s overridden_pharmacy=%.2f "
                "hospital_bill=%.2f total_bill=%.2f",
                baseline_los, expected_total_los, los_delta,
                room_per_day or 0.0, doctor_per_day or 0.0,
                predicted_comp, predicted_pharmacy, comp, pharmacy_final, hospital_final, total_final,
            )
        elif room_per_day is not None and room_per_day > 0:
            predicted_room = float(comp.get("room", 0))
            system_room_total = room_per_day * expected_total_los
            comp["room"] = system_room_total
            hospital_final = hospital_final - predicted_room + system_room_total
            total_final = hospital_final + pharmacy_final
            logger.info(
                "system_room_pricing: expected_total_los_days=%.2f room_per_day=%.2f "
                "predicted_room=%.2f -> overridden_room=%.2f hospital_bill=%.2f total_bill=%.2f",
                expected_total_los, room_per_day, predicted_room, system_room_total,
                hospital_final, total_final,
            )
    elif room_per_day is not None and room_per_day > 0 and expected_total_los > 0:
        predicted_room = float(comp.get("room", 0))
        system_room_total = room_per_day * expected_total_los
        comp["room"] = system_room_total
        hospital_final = hospital_final - predicted_room + system_room_total
        total_final = hospital_final + pharmacy_final
        logger.info(
            "system_room_pricing: expected_total_los_days=%.2f room_per_day=%.2f "
            "predicted_room=%.2f -> overridden_room=%.2f hospital_bill=%.2f total_bill=%.2f",
            expected_total_los, room_per_day, predicted_room, system_room_total,
            hospital_final, total_final,
        )

    def make_component(key: str) -> BillComponent:
        current = float(current_bills.get(key, 0))
        expected = float(comp.get(key, 0))
        return BillComponent(
            current=current,
            expected_final=expected,
            expected_remaining=max(expected - current, 0),
        )

    return EstimateResponse(
        estimate_id=estimate_id,
        estimation_source=estimation_source,
        as_of=datetime.now(IST).isoformat(),
        confidence_band=band,
        confidence_score=float(prediction.get("confidence_score", 0)),
        total_bill=BillComponent(
            current=current_total,
            expected_final=total_final,
            expected_remaining=max(total_final - current_total, 0),
        ),
        hospital_bill=BillComponent(
            current=current_hospital,
            expected_final=hospital_final,
            expected_remaining=max(hospital_final - current_hospital, 0),
        ),
        pharmacy_bill=BillComponent(
            current=current_pharmacy,
            expected_final=pharmacy_final,
            expected_remaining=max(pharmacy_final - current_pharmacy, 0),
        ),
        component_bills=ComponentBills(
            room=make_component("room"),
            doctor=make_component("doctor"),
            laboratory=make_component("laboratory"),
            radiology=make_component("radiology"),
            operating_theatre=make_component("operating_theatre"),
            dialysis=make_component("dialysis"),
            physiotherapy=make_component("physiotherapy"),
            miscellaneous=make_component("miscellaneous"),
        ),
        expected_total_los_days=expected_total_los,
        expected_remaining_los_days=expected_remaining_los,
    )


def _log_fetched_patient(context: str, hr_no: str | None, patient: dict) -> None:
    logger.info(
        "[%s] patient fetched: hr_no=%s ip_id=%s dept=%s doctor=%s room=%s payment=%s "
        "admission_type=%s admission_date=%s diagnosis=%r",
        context, hr_no, patient.get("ip_id"), patient.get("department"), patient.get("doctor_name"),
        patient.get("room_type"), patient.get("payment_type"), patient.get("admission_type"),
        patient.get("admission_date"), patient.get("diagnosis"),
    )


def _log_current_bills(context: str, current_bills: dict) -> None:
    total = sum(float(v) for v in current_bills.values()) if current_bills else 0.0
    logger.info("[%s] current_bills=%s total=%.2f", context, current_bills, total)


def _log_utilization(context: str, utilization: dict) -> None:
    logger.info("[%s] utilization_so_far=%s", context, utilization)


def _log_similar_cases(context: str, similar_cases: list[dict], diagnosis_matched: bool, band: str) -> None:
    logger.info(
        "[%s] find_similar_cases: %d case(s) found, diagnosis_matched=%s, confidence_band=%s",
        context, len(similar_cases), diagnosis_matched, band,
    )
    for c in similar_cases:
        logger.info(
            "[%s]   candidate ip_id=%s match_score=%s los_days=%s total_bill=%s diagnosis_matched=%s",
            context, c.get("ip_id"), c.get("match_score"), c.get("los_days"),
            c.get("bills", {}).get("total"), c.get("diagnosis_matched", True),
        )


def _log_prediction(context: str, label: str, prediction: dict) -> None:
    logger.info(
        "[%s] %s: confidence_score=%s total_bill=%s hospital_bill=%s pharmacy_bill=%s "
        "expected_total_los_days=%s component_bills=%s",
        context, label, prediction.get("confidence_score"), prediction.get("total_bill"),
        prediction.get("hospital_bill"), prediction.get("pharmacy_bill"),
        prediction.get("expected_total_los_days"), prediction.get("component_bills"),
    )


async def estimate_by_hr_no(hr_no: str, prediction_context: str) -> EstimateResponse:
    estimate_id = str(uuid.uuid4())
    ctx = f"by_hr_no:{estimate_id[:8]}"
    logger.info("[%s] request: hr_no=%s prediction_context=%s", ctx, hr_no, prediction_context)

    patient = await fetch_patient_by_hr_no(hr_no)
    if not patient:
        logger.warning("[%s] no admission found for hr_no=%s", ctx, hr_no)
        raise ValueError(f"No admission found for HR number: {hr_no}")
    _log_fetched_patient(ctx, hr_no, patient)
    room_per_day = await fetch_room_rate(patient.get("room_type"), patient.get("payment_type", "cash"))
    doctor_per_day = await fetch_doctor_round_rate(patient.get("doctor_name"))
    logger.info("[%s] room_rate_lookup: room_type=%r payment_type=%r room_per_day=%.2f", ctx, patient.get("room_type"), patient.get("payment_type"), room_per_day)

    current_bills = await fetch_current_bills(patient["ip_id"])
    _log_current_bills(ctx, current_bills)
    utilization = await fetch_admission_utilization(patient["ip_id"], hr_no)
    _log_utilization(ctx, utilization)
    similar_cases = await find_similar_cases(patient, limit=SIMILAR_CASES_ANCHOR_LIMIT)

    diagnosis_matched = any(c.get("diagnosis_matched", True) for c in similar_cases) if similar_cases else True
    band = _confidence_band(len(similar_cases), diagnosis_matched)
    _log_similar_cases(ctx, similar_cases, diagnosis_matched, band)
    # Short suffix: estimates.estimation_source is VARCHAR(20) in the dest DB,
    # and this can combine with the existing "_fallback" suffix below.
    estimation_source = "hr_no" if diagnosis_matched else "hr_no_pd"

    if not similar_cases:
        logger.warning("[%s] no similar cases at all -> returning empty response", ctx)
        response = _empty_response(estimate_id, estimation_source)
        await save_estimate(
            estimate_id, estimation_source,
            {"hr_no": hr_no, "prediction_context": prediction_context},
            response.model_dump(), hr_no=hr_no, ip_id=patient["ip_id"],
        )
        return response

    admission_date = patient.get("admission_date")
    los_days_elapsed = (datetime.now(IST).date() - admission_date).days if admission_date else 0.0
    current_total = sum(float(v) for v in current_bills.values())
    logger.info("[%s] los_days_elapsed=%s current_total=%.2f", ctx, los_days_elapsed, current_total)

    historical_doctor_per_day = await _calculate_doctor_per_day_from_history(similar_cases)
    if historical_doctor_per_day:
        master_doctor_rate = doctor_per_day
        doctor_per_day = historical_doctor_per_day
        logger.info("[%s] doctor_per_day overridden from historical data: %.2f/day (master rate: %.2f)",
                   ctx, doctor_per_day, master_doctor_rate)

    prior_estimate = await fetch_recent_estimate_by_ip_id(patient["ip_id"])
    if prior_estimate is not None:
        logger.info(
            "[%s] prior quote found for this admission: total_bill=%s, expected_total_los_days=%s "
            "-> will clamp this call's result to +/-%.0f%% of both",
            ctx, prior_estimate.get("total_bill", {}).get("expected_final"),
            prior_estimate.get("expected_total_los_days"), HR_NO_RETRIGGER_BAND * 100,
        )

    intensity_ratio = await _intensity_ratio(patient["ip_id"], los_days_elapsed, similar_cases)
    anchor = _similarity_anchor(
        similar_cases, current_total=current_total, elapsed_days=los_days_elapsed,
        intensity_ratio=intensity_ratio,
    )
    logger.info("[%s] computed anchor=%s (band=+/-%.0f%%)", ctx, anchor, ANCHOR_BAND * 100)
    prompt = build_estimation_prompt(
        patient, similar_cases[:SIMILAR_CASES_PROMPT_LIMIT], current_bills,
        utilization=utilization, anchor_total=anchor, anchor_band=ANCHOR_BAND,
    )
    logger.debug("[%s] prompt sent to Gemini (%d chars):\n%s", ctx, len(prompt), prompt)
    try:
        prediction = await get_estimation(prompt)
        _log_prediction(ctx, "raw Gemini prediction", prediction)
    except Exception as e:
        logger.error("[%s] Gemini estimation failed (%s) -> using deterministic peer-average fallback", ctx, e)
        prediction = _peer_average_fallback(similar_cases, current_bills, anchor)
        _log_prediction(ctx, "deterministic fallback prediction", prediction)
        estimation_source = f"{estimation_source}_fallback"

    # Clamp LOS against historical distribution (sanity check)
    diagnosis = patient.get("diagnosis") or ""
    prediction = await _clamp_los_against_historical(prediction, similar_cases, diagnosis, ctx)
    _log_prediction(ctx, "after LOS sanity check", prediction)

    # Resolve component eligibility and enforce it upfront
    component_support = await fetch_diagnosis_component_support(diagnosis)
    eligibility = resolve_component_eligibility(diagnosis, utilization, component_support)
    logger.info("[%s] resolved component eligibility: %s", ctx, eligibility)

    prediction = enforce_component_eligibility(prediction, eligibility, ctx)
    _log_prediction(ctx, "after enforce_component_eligibility (initial)", prediction)

    prediction = _clamp_to_anchor(prediction, anchor)
    _log_prediction(ctx, "final prediction after clamp", prediction)

    if prior_estimate is not None:
        # A prior quote below what's already been billed (current_total) is
        # stale, not a contradiction to enforce — floor the retrigger anchor
        # at current_total the same way expected_total_los is floored at
        # elapsed_days below.
        prior_total = prior_estimate.get("total_bill", {}).get("expected_final")
        if prior_total:
            retrigger_anchor = max(prior_total, current_total)
            prediction = _clamp_to_anchor(prediction, retrigger_anchor, band=HR_NO_RETRIGGER_BAND)
            _log_prediction(ctx, "final prediction after retrigger clamp", prediction)

        prior_los = prior_estimate.get("expected_total_los_days")
        if prior_los:
            retrigger_los_anchor = max(prior_los, los_days_elapsed)
            raw_los = float(prediction.get("expected_total_los_days", 0))
            clamped_los = _clamp_los_to_anchor(raw_los, retrigger_los_anchor, HR_NO_RETRIGGER_BAND)
            if clamped_los != raw_los:
                logger.warning(
                    "[%s] retrigger LOS guard: expected_total_los_days=%.2f outside +/-%.0f%% of prior "
                    "quote's %.2f -> clamped to %.2f",
                    ctx, raw_los, HR_NO_RETRIGGER_BAND * 100, retrigger_los_anchor, clamped_los,
                )
            prediction = dict(prediction)
            prediction["expected_total_los_days"] = clamped_los

    # Enforce component eligibility: zero out components that don't belong to this diagnosis
    prediction = enforce_component_eligibility(prediction, eligibility, ctx)
    _log_prediction(ctx, "after enforce_component_eligibility", prediction)

    response = _build_response(
        estimate_id, estimation_source, band, current_bills, prediction,
        los_days_elapsed=los_days_elapsed, room_per_day=room_per_day,
        doctor_per_day=doctor_per_day, similar_cases=similar_cases,
        component_support=component_support, eligibility=eligibility,
    )
    await save_estimate(
        estimate_id, estimation_source,
        {"hr_no": hr_no, "prediction_context": prediction_context},
        response.model_dump(), hr_no=hr_no, ip_id=patient["ip_id"],
    )
    logger.info(
        "[%s] response: estimation_source=%s confidence_band=%s total_bill.expected_final=%.2f",
        ctx, estimation_source, band, response.total_bill.expected_final,
    )
    return response


def _peer_average_fallback(
    similar_cases: list[dict], current_bills: dict, anchor: float | None, target_los: float | None = None,
) -> dict:
    """Deterministic prediction used when Gemini itself fails outright (timeout,
    API error, no readable/parseable output after retry) on a non-override
    estimate — distributes the already-computed anchor total across
    components using each peer case's own bucket share (same match_score
    weighting as _similarity_anchor), so a Gemini outage 500s the two most
    common endpoints (by-hr-no and initial) instead of returning nothing.
    """
    bucket_keys = BILL_KEYS + ["pharmacy"]
    weighted_share = {k: 0.0 for k in bucket_keys}
    weighted_los_sum = 0.0
    weight_total = 0.0
    for case in similar_cases:
        los = case.get("los_days") or 0
        if not los or los <= 0:
            continue
        bills = case.get("bills", {})
        total = bills.get("total") or sum(bills.get(k, 0) for k in bucket_keys)
        if not total:
            continue
        weight = float(case.get("match_score") or 0) + 1
        for k in bucket_keys:
            weighted_share[k] += weight * (bills.get(k, 0) / total)
        weighted_los_sum += weight * los
        weight_total += weight

    current_total = sum(float(v) for v in current_bills.values()) if current_bills else 0.0
    if weight_total == 0:
        # No peer had a usable bucket breakdown either -- last resort: split
        # the anchor evenly rather than fail with no estimate at all.
        share = {k: 1 / len(bucket_keys) for k in bucket_keys}
        expected_los = target_los or 0
    else:
        share_sum = sum(weighted_share.values()) or 1.0
        share = {k: v / share_sum for k, v in weighted_share.items()}
        expected_los = target_los if target_los is not None else weighted_los_sum / weight_total

    total_bill = anchor if anchor else current_total
    component_bills = {k: total_bill * share.get(k, 0) for k in BILL_KEYS}
    pharmacy_bill = total_bill * share.get("pharmacy", 0)
    hospital_bill = sum(component_bills.values())

    logger.info(
        "peer_average_fallback: weight_total=%.2f total_bill=%.2f hospital_bill=%.2f pharmacy_bill=%.2f "
        "expected_total_los_days=%.2f",
        weight_total, total_bill, hospital_bill, pharmacy_bill, expected_los,
    )

    return {
        "confidence_score": 20,
        "total_bill": hospital_bill + pharmacy_bill,
        "hospital_bill": hospital_bill,
        "pharmacy_bill": pharmacy_bill,
        "component_bills": component_bills,
        "expected_total_los_days": expected_los,
    }


DOCTOR_OVERRIDE_PERCENT = 0.25
LAB_OVERRIDE_PERCENT = 0.25
PHARMACY_OVERRIDE_PERCENT = 0.5


async def _los_override_fallback(patient: dict, similar_cases: list[dict], new_los_days: float) -> dict:
    """Deterministic Additional Day Cost estimate, used only when Gemini fails to
    return a usable prediction for a LOS-override request:

        Additional Day Cost = Room/day + partial Doctor/day
            + 50% of Average Pharmacy/day + 25% of Average Laboratory/day
            + LOS-dependent Other charges/day (physiotherapy + miscellaneous)

    Component LOS behavior:
        Room              -> fully LOS-sensitive
        Doctor            -> mostly fixed, lightly LOS-sensitive
        OT / radiology    -> fixed
        Lab               -> partially LOS-sensitive
        Pharmacy          -> partially LOS-sensitive
    """
    valid_cases = [c for c in similar_cases if c.get("los_days") and c["los_days"] > 0]
    baseline_los = median([c["los_days"] for c in valid_cases]) if valid_cases else 1
    additional_days = max(new_los_days - baseline_los, 0)

    def bucket_avg(key: str) -> float:
        vals = [c["bills"].get(key, 0) for c in valid_cases]
        return sum(vals) / len(vals) if vals else 0.0

    def bucket_per_day_avg(key: str) -> float:
        vals = [c["bills"].get(key, 0) / c["los_days"] for c in valid_cases]
        return sum(vals) / len(vals) if vals else 0.0

    room_per_day = await fetch_room_rate(patient.get("room_type"), patient.get("payment_type", "cash"))
    doctor_per_day = await fetch_doctor_round_rate(patient.get("doctor_name"))
    pharmacy_per_day = bucket_per_day_avg("pharmacy")
    lab_per_day = bucket_per_day_avg("laboratory")
    physio_per_day = bucket_per_day_avg("physiotherapy")
    misc_per_day = bucket_per_day_avg("miscellaneous")

    additional_day_cost = (
        room_per_day
        + DOCTOR_OVERRIDE_PERCENT * doctor_per_day
        + PHARMACY_OVERRIDE_PERCENT * pharmacy_per_day
        + LAB_OVERRIDE_PERCENT * lab_per_day
        + physio_per_day
        + misc_per_day
    )

    component_bills = {
        "room": bucket_avg("room") + room_per_day * additional_days,
        "doctor": bucket_avg("doctor") + DOCTOR_OVERRIDE_PERCENT * doctor_per_day * additional_days,
        "laboratory": bucket_avg("laboratory") + LAB_OVERRIDE_PERCENT * lab_per_day * additional_days,
        "radiology": bucket_avg("radiology"),
        "operating_theatre": bucket_avg("operating_theatre"),
        "dialysis": bucket_avg("dialysis"),
        "physiotherapy": bucket_avg("physiotherapy") + physio_per_day * additional_days,
        "pharmacy": bucket_avg("pharmacy") + PHARMACY_OVERRIDE_PERCENT * pharmacy_per_day * additional_days,
        "miscellaneous": bucket_avg("miscellaneous") + misc_per_day * additional_days,
    }

    pharmacy_bill = component_bills["pharmacy"]
    hospital_bill = sum(v for k, v in component_bills.items() if k != "pharmacy")

    logger.info(
        "los_override_fallback: baseline_los=%.2f (from %d valid peer cases) additional_days=%.2f "
        "room_per_day=%.2f doctor_per_day=%.2f additional_day_cost=%.2f -> hospital_bill=%.2f pharmacy_bill=%.2f",
        baseline_los, len(valid_cases), additional_days, room_per_day, doctor_per_day,
        additional_day_cost, hospital_bill, pharmacy_bill,
    )

    return {
        "confidence_score": 30,
        "total_bill": hospital_bill + pharmacy_bill,
        "hospital_bill": hospital_bill,
        "pharmacy_bill": pharmacy_bill,
        "component_bills": component_bills,
        "additional_day_cost": additional_day_cost,
        "additional_days": additional_days,
    }


def _manual_estimate_signature(
    patient: dict, los_override_days: float | None, old_los: float | None, old_estimate: float | None,
) -> str:
    """Hash of the fields that actually drive estimate_manual's computation
    (peer selection + anchor). Deliberately excludes comments/override reason
    text, which are descriptive only and don't affect the prediction, so a
    clinician's note doesn't defeat caching for an otherwise identical case.
    """
    signature_fields = {
        "patient": patient,
        "los_override_days": los_override_days,
        "old_los": old_los,
        "old_estimate": old_estimate,
    }
    return hashlib.sha256(json.dumps(signature_fields, sort_keys=True, default=str).encode()).hexdigest()


def _manual_profile_signature(
    patient: dict, los_override_days: float | None, old_los: float | None, old_estimate: float | None,
) -> str:
    """Same as _manual_estimate_signature but with doctor_name excluded — used
    to find the last quote for the SAME case regardless of doctor, so a
    doctor-only change can be clamped against it (see DOCTOR_CHANGE_BAND).
    """
    profile_patient = {k: v for k, v in patient.items() if k != "doctor_name"}
    return _manual_estimate_signature(profile_patient, los_override_days, old_los, old_estimate)


def _normalize_manual_patient(payload: InitialEstimateRequest) -> dict:
    return {
        "patient_name": payload.patient_name,
        "gender": payload.gender,
        "age": payload.age,
        "department": payload.department,
        "doctor_name": payload.doctor_name,
        "room_type": payload.room_type,
        "payment_type": payload.payment_method,
        "diagnosis": payload.diagnosis,
        "admission_type": payload.admission_type,
    }


async def estimate_manual(
    payload: InitialEstimateRequest,
    estimation_source: str = "initial",
    los_override_days: float | None = None,
    los_override_reason: str | None = None,
    old_los: float | None = None,
    old_estimate: float | None = None,
) -> EstimateResponse:
    estimate_id = str(uuid.uuid4())
    ctx = f"manual:{estimate_id[:8]}"
    logger.info(
        "[%s] request: diagnosis=%r dept=%s doctor=%s room=%s payment=%s age=%s gender=%s "
        "admission_type=%s los_override_days=%s los_override_reason=%r old_los=%s old_estimate=%s",
        ctx, payload.diagnosis, payload.department, payload.doctor_name, payload.room_type,
        payload.payment_method, payload.age, payload.gender, payload.admission_type,
        los_override_days, los_override_reason, old_los, old_estimate,
    )
    patient = _normalize_manual_patient(payload)
    current_bills: dict = {}
    room_per_day = await fetch_room_rate(patient.get("room_type"), patient.get("payment_type", "cash"))
    doctor_per_day = await fetch_doctor_round_rate(patient.get("doctor_name"))
    logger.info("[%s] room_rate_lookup: room_type=%r payment_type=%r room_per_day=%.2f", ctx, patient.get("room_type"), patient.get("payment_type"), room_per_day)

    request_payload = payload.model_dump()
    if los_override_days is not None:
        request_payload["new_los_days"] = los_override_days
        request_payload["reason"] = los_override_reason
        request_payload["old_los"] = old_los
        request_payload["old_estimate"] = old_estimate

    signature = _manual_estimate_signature(patient, los_override_days, old_los, old_estimate)
    profile_signature = _manual_profile_signature(patient, los_override_days, old_los, old_estimate)

    cached = await fetch_recent_estimate_by_signature(signature, MANUAL_ESTIMATE_CACHE_TTL_HOURS)
    if cached is not None:
        logger.info(
            "[%s] identical input seen within %dh (signature=%s) -> returning cached estimate instead of recomputing",
            ctx, MANUAL_ESTIMATE_CACHE_TTL_HOURS, signature[:12],
        )
        cached["estimate_id"] = estimate_id
        cached["as_of"] = datetime.now(IST).isoformat()
        response = EstimateResponse(**cached)
        await save_estimate(
            estimate_id, response.estimation_source, request_payload, response.model_dump(),
            request_signature=signature, profile_signature=profile_signature,
        )
        return response

    doctor_change_anchor = None
    doctor_change_los_anchor = None
    profile_match = await fetch_recent_estimate_by_profile_signature(profile_signature, MANUAL_ESTIMATE_CACHE_TTL_HOURS)
    if profile_match is not None:
        prior_doctor = profile_match["request_payload"].get("doctor_name")
        if prior_doctor and prior_doctor != payload.doctor_name:
            doctor_change_anchor = profile_match["response_payload"].get("total_bill", {}).get("expected_final")
            doctor_change_los_anchor = profile_match["response_payload"].get("expected_total_los_days")
            logger.info(
                "[%s] same case previously quoted for doctor=%r at total_bill=%s, expected_total_los_days=%s "
                "-> will clamp doctor=%r result to +/-%.0f%% of both",
                ctx, prior_doctor, doctor_change_anchor, doctor_change_los_anchor,
                payload.doctor_name, DOCTOR_CHANGE_BAND * 100,
            )

    similar_cases = await find_similar_cases(patient, limit=SIMILAR_CASES_ANCHOR_LIMIT)
    diagnosis_matched = any(c.get("diagnosis_matched", True) for c in similar_cases) if similar_cases else True
    band = _confidence_band(len(similar_cases), diagnosis_matched)
    _log_similar_cases(ctx, similar_cases, diagnosis_matched, band)
    if not diagnosis_matched:
        # Short suffix: estimates.estimation_source is VARCHAR(20) in the dest
        # DB, and this can combine with the "_fallback" suffix below.
        estimation_source = f"{estimation_source}_pd"

    historical_doctor_per_day = await _calculate_doctor_per_day_from_history(similar_cases)
    if historical_doctor_per_day:
        master_doctor_rate = doctor_per_day
        doctor_per_day = historical_doctor_per_day
        logger.info("[%s] doctor_per_day overridden from historical data: %.2f/day (master rate: %.2f)",
                   ctx, doctor_per_day, master_doctor_rate)

    if not similar_cases:
        logger.warning("[%s] no similar cases at all -> returning empty response", ctx)
        response = _empty_response(estimate_id, estimation_source)
        await save_estimate(
            estimate_id, estimation_source, request_payload, response.model_dump(),
            request_signature=signature, profile_signature=profile_signature,
        )
        return response

    if los_override_days is not None and old_los is not None and old_estimate is not None:
        anchor = _override_anchor(old_los, old_estimate, los_override_days)
    else:
        anchor = _similarity_anchor(similar_cases, target_los=los_override_days)
    clamp_band = LOS_OVERRIDE_ANCHOR_BAND if los_override_days is not None else ANCHOR_BAND
    logger.info("[%s] computed anchor=%s (band=+/-%.0f%%)", ctx, anchor, clamp_band * 100)
    prompt = build_estimation_prompt(
        patient, similar_cases[:SIMILAR_CASES_PROMPT_LIMIT], current_bills,
        los_override_days=los_override_days,
        los_override_reason=los_override_reason,
        anchor_total=anchor,
        anchor_band=clamp_band,
    )
    logger.debug("[%s] prompt sent to Gemini (%d chars):\n%s", ctx, len(prompt), prompt)

    try:
        prediction = await get_estimation(prompt)
        _log_prediction(ctx, "raw Gemini prediction", prediction)
    except Exception as e:
        logger.error("[%s] Gemini estimation failed (%s) -> using deterministic fallback", ctx, e)
        if los_override_days is None:
            prediction = _peer_average_fallback(similar_cases, current_bills, anchor)
        else:
            prediction = await _los_override_fallback(patient, similar_cases, los_override_days)
        _log_prediction(ctx, "deterministic fallback prediction", prediction)
        estimation_source = f"{estimation_source}_fallback"

    # Clamp LOS against historical distribution (sanity check)
    # Skip if this is a manual LOS override (user already decided the LOS)
    if los_override_days is None:
        prediction = await _clamp_los_against_historical(prediction, similar_cases, payload.diagnosis, ctx)
        _log_prediction(ctx, "after LOS sanity check", prediction)

    # Resolve component eligibility and enforce it upfront
    # Note: estimate_manual has no live admission, so utilization is None (diagnosis-only evidence)
    component_support = await fetch_diagnosis_component_support(payload.diagnosis)
    eligibility = resolve_component_eligibility(payload.diagnosis, utilization=None, component_support=component_support)
    logger.info("[%s] resolved component eligibility: %s", ctx, eligibility)

    prediction = enforce_component_eligibility(prediction, eligibility, ctx)
    _log_prediction(ctx, "after enforce_component_eligibility (initial)", prediction)

    prediction = _clamp_to_anchor(prediction, anchor, band=clamp_band)
    _log_prediction(ctx, "final prediction after clamp", prediction)

    if doctor_change_anchor:
        prediction = _clamp_to_anchor(prediction, doctor_change_anchor, band=DOCTOR_CHANGE_BAND)
        _log_prediction(ctx, "final prediction after doctor-change clamp", prediction)

    # Enforce component eligibility: zero out components that don't belong to this diagnosis
    prediction = enforce_component_eligibility(prediction, eligibility, ctx)
    _log_prediction(ctx, "after enforce_component_eligibility", prediction)

    # LOS is predicted independently of total_bill, so a bill landing inside
    # its band doesn't stop Gemini from swinging the stay length itself (an
    # 8-day vs. 16-day prediction for the same case, doctor aside, at a
    # similar total). Doesn't apply to LOS-override calls: expected_total_los
    # there is the clinician's own override value, not a Gemini prediction.
    if doctor_change_los_anchor and los_override_days is None:
        raw_los = float(prediction.get("expected_total_los_days", 0))
        clamped_los = _clamp_los_to_anchor(raw_los, doctor_change_los_anchor, DOCTOR_CHANGE_BAND)
        if clamped_los != raw_los:
            logger.warning(
                "[%s] doctor-change LOS guard: expected_total_los_days=%.2f outside +/-%.0f%% of prior "
                "quote's %.2f -> clamped to %.2f",
                ctx, raw_los, DOCTOR_CHANGE_BAND * 100, doctor_change_los_anchor, clamped_los,
            )
        prediction = dict(prediction)
        prediction["expected_total_los_days"] = clamped_los

    response = _build_response(
        estimate_id, estimation_source, band, current_bills, prediction,
        los_override_days=los_override_days, room_per_day=room_per_day,
        doctor_per_day=doctor_per_day, similar_cases=similar_cases,
        component_support=component_support, eligibility=eligibility,
    )
    await save_estimate(
        estimate_id, estimation_source, request_payload, response.model_dump(),
        request_signature=signature, profile_signature=profile_signature,
    )
    logger.info(
        "[%s] response: estimation_source=%s confidence_band=%s total_bill.expected_final=%.2f",
        ctx, estimation_source, band, response.total_bill.expected_final,
    )
    return response

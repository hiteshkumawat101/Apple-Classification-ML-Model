"""
test_estimation_api.py — Estimation API test with realistic inputs
Usage: python3 test_estimation_api.py

DB reference figures (median/n) are fetched live from the source DB via the
same find_similar_cases() logic the API itself uses, instead of hardcoded
numbers — those had drifted from what the DB (and the now-fixed matching
logic) actually returns.
"""
import asyncio
import json
import http.client
from statistics import median

from app.db.connection import create_pool, close_pool
from app.db.queries import find_similar_cases

HOST = "127.0.0.1:8003"
ENDPOINT = "/v1/estimates/initial"

# Realistic inputs derived from actual DB cases
# (doctor, room, dept, age, gender, payment) all from real admissions
TESTS = [
    # diagnosis, dept, age, gender, doctor, room, payment, mobile
    ("Total Knee Replacement",            "Orthopaedics",     58, "Male",   "Dr. Subramanian.M", "General Ward", "Cash", "9700000001"),
    ("Hysterectomy",                      "Gynaecology",      45, "Female", "Dr. Vijayan J",       "Single Room",  "Cash", "9700000002"),
    ("Cesarean Section",                  "Gynaecology",      28, "Female", "Dr. Abinaya Maathuri", "Single Room",  "Cash", "9700000003"),
    ("Laparoscopic Cholecystectomy",      "Surgery",          45, "Female", "Dr. Ilavarasan S",    "General Ward", "Cash", "9700000004"),
    ("Appendectomy",                      "Surgery",          28, "Male",   "Dr. Ilavarasan S",    "General Ward", "Cash", "9700000005"),
    ("Hernia Repair",                     "Surgery",          40, "Male",   "Dr. Ilavarasan S",    "General Ward", "Cash", "9700000006"),
    ("Fracture Femur with ORIF",          "Orthopaedics",     45, "Male",   "Dr. Subramanian.M",   "Single Room",  "Cash", "9700000007"),
    ("Acute Ischemic Stroke",             "Neurology",        65, "Male",   "Dr. Ranjiith",        "General Ward", "Cash", "9700000008"),
    ("Acute Myocardial Infarction",       "Cardiology",       60, "Male",   "Dr. Mahin",           "General Ward", "Cash", "9700000009"),
    ("Dengue Fever with Warning Signs",   "General Medicine", 35, "Male",   "Dr. Sarita Vinod",    "General Ward", "Cash", "9700000010"),
    ("Severe Pneumonia",                  "General Medicine", 70, "Male",   "Dr. Sarita Vinod",    "General Ward", "Cash", "9700000011"),
    ("Normal Vaginal Delivery",           "Gynaecology",      26, "Female", "Dr. Abinaya Maathuri", "General Ward", "Cash", "9700000012"),
    ("Spinal Decompression Surgery",      "Orthopaedics",     50, "Male",   "Dr. Subramanian.M",   "Single Room",  "Cash", "9700000013"),
    ("Total Hip Replacement",             "Orthopaedics",     65, "Male",   "Dr. Subramanian.M",   "Single Room",  "Cash", "9700000014"),
    ("Coronary Angioplasty with Stenting", "Cardiology",      55, "Male",   "Dr. Mahin",           "Single Room",  "Cash", "9700000015"),
    ("Coronary Artery Bypass Grafting",   "Cardiology",       60, "Male",   "Dr. Mahin",           "Single Room",  "Cash", "9700000016"),
]


async def fetch_db_references() -> dict:
    """For each test diagnosis, pull the same similar-case pool find_similar_cases()
    would use for that request, and derive a live median/n reference from it."""
    await create_pool()
    refs = {}
    try:
        for diag, dept, age, gender, doctor, room, payment, mobile in TESTS:
            patient = {
                "diagnosis": diag,
                "department": dept,
                "gender": gender,
                "age": age,
                "doctor_name": doctor,
                "room_type": room,
            }
            cases = await find_similar_cases(patient, limit=25)
            totals = [c["bills"]["total"] for c in cases if c.get("bills")]
            refs[diag] = (round(median(totals), 2), len(totals)) if totals else (0, 0)
    finally:
        await close_pool()
    return refs


def call(diag, dept, age, gender, doctor, room, payment, mobile):
    payload = json.dumps({
        "patient_name": "Test",
        "mobile_number": mobile,
        "age": age,
        "diagnosis": diag,
        "doctor_name": doctor,
        "gender": gender,
        "payment_method": payment,
        "department": dept,
        "room_type": room
    })
    c = http.client.HTTPConnection(HOST, timeout=90)
    c.request("POST", ENDPOINT, body=payload,
              headers={"Content-Type": "application/json"})
    resp = c.getresponse()
    raw = resp.read().decode()
    c.close()
    return json.loads(raw) if raw.strip() else None


def run():
    SEP = "=" * 100

    print(SEP)
    print("ESTIMATION API — FULL TEST WITH REALISTIC INPUTS")
    print("(doctor + room derived from actual DB admissions; DB reference fetched live)")
    print(SEP)

    print("Fetching live DB reference figures (find_similar_cases per diagnosis)...")
    db_refs = asyncio.run(fetch_db_references())

    print(f"\n  {'Diagnosis':<40} {'API Estimate':>13} {'DB Median':>11} {'DB n':>5} {'Ratio':>7} {'Conf':<8} {'Flags'}")
    print(f"  {'-'*100}")

    issues = []

    for diag, dept, age, gender, doctor, room, payment, mobile in TESTS:
        d = call(diag, dept, age, gender, doctor, room, payment, mobile)
        db_med, db_n = db_refs.get(diag, (0, 0))

        if not d or "total_bill" not in d:
            print(f"  {diag:<40} ERROR")
            continue

        est   = d["total_bill"]["expected_final"]
        conf  = d.get("confidence_band", "")
        warns = d.get("warnings", [])

        fl = []
        if any("No matched" in w for w in warns):         fl.append("NO_MATCH")
        if any("Category detection" in w for w in warns): fl.append("CAT_LOW")
        if any("Clamped" in w for w in warns):            fl.append("LOS_CLAMPED")

        if db_med > 0:
            ratio = est / db_med
            ratio_str = f"{ratio:.2f}x"
            if ratio < 0.7:
                verdict = "⚠️  UNDER"
                issues.append((diag, est, db_med, ratio, conf, fl))
            elif ratio > 1.5:
                verdict = "⚠️  OVER"
                issues.append((diag, est, db_med, ratio, conf, fl))
            else:
                verdict = "✓"
        else:
            ratio_str = "N/A"
            verdict = "no DB ref"
            if "NO_MATCH" in fl:
                issues.append((diag, est, 0, 0, conf, fl))

        db_str  = f"Rs.{db_med:,.0f}" if db_med else "N/A"
        est_str = f"Rs.{est:,.0f}"
        flag_str = ",".join(fl) if fl else "ok"

        print(f"  {diag:<40} {est_str:>13} {db_str:>11} {db_n:>5} {ratio_str:>7}  {conf:<8} {flag_str}  {verdict}")

    # Summary
    print(f"\n{SEP}")
    print(f"  ISSUES FOUND: {len(issues)}")
    print(SEP)
    for diag, est, db_med, ratio, conf, fl in issues:
        if db_med > 0:
            print(f"  ⚠️  {diag}")
            print(f"       API=Rs.{est:,.0f}  DB median=Rs.{db_med:,.0f}  ratio={ratio:.2f}x  conf={conf}  flags={fl}")
        else:
            print(f"  ⚠️  {diag}  — no DB history, pure Gemini guess  flags={fl}")
    print(SEP)


if __name__ == "__main__":
    run()

"""Parser tests use real captured COROS MCP text responses (from a live spike
against a real connected account — see docs/tasks/10-coros-sync-layer.md) as
fixtures, so a parsing regression is caught against the real vendor shape,
not a hand-written approximation of it.
"""

from app.services import coros_sync_parsers as parsers

SPORT_RECORDS_TEXT = """Sport Records — 2026-08-27 to 2026-09-24 (3 records)
========================

1. Walk — 2026-09-23
   Location: Pune Walk
   Start Coordinates: 18.511999, 73.857002
   Time Window: startTimestamp=1790187250 | endTimestamp=1790194508
   Duration: 2:00:58 | Distance: 2.85 km
   Average Pace: 42:23 /km | Avg HR: 95 bpm | Calories: 652 kcal
   LabelId: 480549930019095028 | SportType: 900

2. Strength — 2026-09-22
   Location: Strength
   Time Window: startTimestamp=1790086962 | endTimestamp=1790088182
   Duration: 20:19 | Sets: 1
 | Avg HR: 97 bpm | Calories: 136 kcal
   LabelId: 480523140972183554 | SportType: 402

3. Outdoor Run — 2026-09-21
   Location: Pune Run
   Start Coordinates: 18.563000, 73.754997
   Time Window: startTimestamp=1789996097 | endTimestamp=1789997859
   Duration: 29:22 | Distance: 4.02 km
   Average Pace: 7:18 /km | Avg HR: 151 bpm | Calories: 453 kcal
   LabelId: 480498903391698948 | SportType: 100
"""

RESTING_HR_TEXT = """Resting Heart Rate — Last 4 days
========================

2026-09-24: No data
2026-09-23: 53 bpm
2026-09-22: 52 bpm
2026-09-21: 49 bpm"""

AVG_HR_TEXT = """Average Heart Rate — Last 2 days
========================

2026-09-24: 66 bpm (Min: 45, Max: 105)
2026-09-23: 62 bpm (Min: 37, Max: 104)"""

TRAINING_LOAD_TEXT = """Training Load Assessment
========================

2026-09-24
Comment: Resuming
Short-Term Load: 22
Long-Term Load: 40
Load Ratio: 0.55

2026-09-23
Comment: Decreasing
Short-Term Load: 26
Long-Term Load: 41
Load Ratio: 0.63"""

RECOVERY_TEXT = """Recovery Status
========================

Recovery: 100%
Level: Heavy training allowed
Estimated Full Recovery: 0h"""

FITNESS_TEXT = """Fitness Assessment Overview
========================

VO2max: 49
Running Level: 70
Threshold Pace: 5:32 /km
5 km Prediction: 26:40
10 km Prediction: 55:59
Half Marathon Prediction: 2:06:06
Marathon Prediction: 4:27:29"""


def test_parse_sport_records_extracts_all_blocks():
    activities = parsers.parse_sport_records(SPORT_RECORDS_TEXT)
    assert len(activities) == 3


def test_parse_sport_records_maps_disciplines_correctly():
    activities = parsers.parse_sport_records(SPORT_RECORDS_TEXT)
    by_id = {a.external_id: a for a in activities}
    assert by_id["480498903391698948"].discipline == "run"  # SportType 100
    assert by_id["480549930019095028"].discipline == "other"  # SportType 900 (walk)
    assert by_id["480523140972183554"].discipline == "other"  # SportType 402 (strength)


def test_parse_sport_records_handles_missing_distance_pace():
    activities = parsers.parse_sport_records(SPORT_RECORDS_TEXT)
    strength = next(a for a in activities if a.external_id == "480523140972183554")
    assert strength.distance_km is None
    assert strength.avg_pace_sec_per_km is None
    assert strength.avg_hr == 97
    assert strength.calories == 136


def test_parse_sport_records_parses_numeric_fields():
    activities = parsers.parse_sport_records(SPORT_RECORDS_TEXT)
    run = next(a for a in activities if a.external_id == "480498903391698948")
    assert run.duration_seconds == 29 * 60 + 22
    assert run.distance_km == 4.02
    assert run.avg_pace_sec_per_km == 7 * 60 + 18
    assert run.avg_hr == 151
    assert run.calories == 453
    assert run.started_at_epoch == 1789996097
    assert run.ended_at_epoch == 1789997859
    assert run.sport_type_code == 100


def test_parse_resting_heart_rate():
    values = parsers.parse_resting_heart_rate(RESTING_HR_TEXT)
    assert len(values) == 4
    by_date = {v.date: v.value for v in values}
    assert by_date["2026-09-24"] is None  # "No data"
    assert by_date["2026-09-23"] == 53.0


def test_parse_avg_heart_rate_includes_min_max():
    values = parsers.parse_avg_heart_rate(AVG_HR_TEXT)
    assert len(values) == 2
    first = values[0]
    assert first.date == "2026-09-24"
    assert first.value == 66.0
    assert first.extra == {"min": 45, "max": 105}


def test_parse_training_load_assessment():
    values = parsers.parse_training_load_assessment(TRAINING_LOAD_TEXT)
    assert len(values) == 2
    latest = values[0]
    assert latest.date == "2026-09-24"
    assert latest.value == 0.55  # load ratio is the primary value
    assert latest.extra == {"comment": "Resuming", "short_term_load": 22.0, "long_term_load": 40.0}


def test_parse_recovery_status():
    data = parsers.parse_recovery_status(RECOVERY_TEXT)
    assert data == {
        "recovery_pct": 100,
        "level": "Heavy training allowed",
        "estimated_full_recovery": "0h",
    }


def test_parse_fitness_assessment_overview():
    data = parsers.parse_fitness_assessment_overview(FITNESS_TEXT)
    assert data["vo2max"] == "49"
    assert data["threshold_pace"] == "5:32 /km"
    assert data["prediction_5k"] == "26:40"


def test_discipline_for_sport_code_covers_run_bike_swim_other():
    assert parsers.discipline_for_sport_code(100) == "run"
    assert parsers.discipline_for_sport_code(203) == "bike"
    assert parsers.discipline_for_sport_code(301) == "swim"
    assert parsers.discipline_for_sport_code(9999) == "other"


def test_parsers_degrade_gracefully_on_unexpected_text():
    """A malformed/empty response shouldn't raise — it should parse to nothing,
    so one drifted tool response doesn't take down the whole sync."""
    assert parsers.parse_sport_records("nonsense") == []
    assert parsers.parse_resting_heart_rate("nonsense") == []
    assert parsers.parse_training_load_assessment("") == []
    assert parsers.parse_recovery_status("nonsense") == {}
    assert parsers.parse_fitness_assessment_overview("") == {}

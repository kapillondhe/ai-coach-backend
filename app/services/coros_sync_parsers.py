
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_RUN_CODES = {100, 101, 102, 103, 104, 105, 106}
_BIKE_CODES = {200, 201, 202, 203, 204, 205, 299}
_SWIM_CODES = {300, 301}


def discipline_for_sport_code(code: int) -> str:
    if code in _RUN_CODES:
        return "run"
    if code in _BIKE_CODES:
        return "bike"
    if code in _SWIM_CODES:
        return "swim"
    return "other"


def _duration_to_seconds(text: str) -> int | None:
    """Parse 'H:MM:SS' or 'MM:SS' into seconds."""
    parts = text.strip().split(":")
    try:
        parts_int = [int(p) for p in parts]
    except ValueError:
        return None
    if len(parts_int) == 3:
        h, m, s = parts_int
        return h * 3600 + m * 60 + s
    if len(parts_int) == 2:
        m, s = parts_int
        return m * 60 + s
    return None


def _pace_to_sec_per_km(text: str) -> int | None:
    """Parse a 'M:SS' or 'MM:SS' pace like '7:18' into seconds/km."""
    match = re.match(r"^(\d+):(\d{2})$", text.strip())
    if not match:
        return None
    minutes, seconds = int(match.group(1)), int(match.group(2))
    return minutes * 60 + seconds


@dataclass
class ParsedActivity:
    external_id: str
    discipline: str
    sport_type_code: int
    started_at_epoch: int | None
    ended_at_epoch: int | None
    duration_seconds: int | None
    distance_km: float | None
    avg_pace_sec_per_km: int | None
    avg_hr: int | None
    calories: int | None
    raw_text: str


_ACTIVITY_BLOCK_RE = re.compile(r"^\d+\.\s+.+? — \d{4}-\d{2}-\d{2}$")
_TIME_WINDOW_RE = re.compile(r"startTimestamp=(\d+)\s*\|\s*endTimestamp=(\d+)")
_DURATION_RE = re.compile(r"Duration:\s*([\d:]+)")
_DISTANCE_RE = re.compile(r"Distance:\s*([\d.]+)\s*km")
_PACE_RE = re.compile(r"Average Pace:\s*([\d:]+)\s*/km")
_HR_RE = re.compile(r"Avg HR:\s*(\d+)\s*bpm")
_CALORIES_RE = re.compile(r"Calories:\s*(\d+)\s*kcal")
_LABEL_SPORT_RE = re.compile(r"LabelId:\s*(\S+)\s*\|\s*SportType:\s*(\d+)")


def parse_sport_records(text: str) -> list[ParsedActivity]:
    """Parse querySportRecords' text into one ParsedActivity per numbered block."""
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if _ACTIVITY_BLOCK_RE.match(stripped):
            if current:
                blocks.append(current)
            current = [stripped]
        elif current:
            current.append(stripped)
    if current:
        blocks.append(current)

    activities: list[ParsedActivity] = []
    for block in blocks:
        block_text = "\n".join(block)
        label_match = _LABEL_SPORT_RE.search(block_text)
        if not label_match:
            logger.warning("Skipping sport-record block with no LabelId/SportType: %r", block_text[:120])
            continue
        external_id, sport_type_code = label_match.group(1), int(label_match.group(2))

        time_match = _TIME_WINDOW_RE.search(block_text)
        duration_match = _DURATION_RE.search(block_text)
        distance_match = _DISTANCE_RE.search(block_text)
        pace_match = _PACE_RE.search(block_text)
        hr_match = _HR_RE.search(block_text)
        cal_match = _CALORIES_RE.search(block_text)

        activities.append(
            ParsedActivity(
                external_id=external_id,
                discipline=discipline_for_sport_code(sport_type_code),
                sport_type_code=sport_type_code,
                started_at_epoch=int(time_match.group(1)) if time_match else None,
                ended_at_epoch=int(time_match.group(2)) if time_match else None,
                duration_seconds=_duration_to_seconds(duration_match.group(1)) if duration_match else None,
                distance_km=float(distance_match.group(1)) if distance_match else None,
                avg_pace_sec_per_km=_pace_to_sec_per_km(pace_match.group(1)) if pace_match else None,
                avg_hr=int(hr_match.group(1)) if hr_match else None,
                calories=int(cal_match.group(1)) if cal_match else None,
                raw_text=block_text,
            )
        )
    return activities


@dataclass
class ParsedDailyValue:
    date: str  # yyyy-mm-dd
    value: float | None
    extra: dict = field(default_factory=dict)


_RESTING_HR_LINE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}):\s*(?:(\d+)\s*bpm|No data)$")


def parse_resting_heart_rate(text: str) -> list[ParsedDailyValue]:
    values = []
    for line in text.splitlines():
        match = _RESTING_HR_LINE_RE.match(line.strip())
        if not match:
            continue
        date, bpm = match.groups()
        values.append(ParsedDailyValue(date=date, value=float(bpm) if bpm else None))
    return values


_AVG_HR_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}):\s*(?:(\d+)\s*bpm\s*\(Min:\s*(\d+),\s*Max:\s*(\d+)\)|No data)$"
)


def parse_avg_heart_rate(text: str) -> list[ParsedDailyValue]:
    values = []
    for line in text.splitlines():
        match = _AVG_HR_LINE_RE.match(line.strip())
        if not match:
            continue
        date, bpm, hr_min, hr_max = match.groups()
        extra = {"min": int(hr_min), "max": int(hr_max)} if hr_min and hr_max else {}
        values.append(ParsedDailyValue(date=date, value=float(bpm) if bpm else None, extra=extra))
    return values


_LOAD_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")
_LOAD_COMMENT_RE = re.compile(r"^Comment:\s*(.+)$")
_LOAD_SHORT_RE = re.compile(r"^Short-Term Load:\s*([\d.]+)$")
_LOAD_LONG_RE = re.compile(r"^Long-Term Load:\s*([\d.]+)$")
_LOAD_RATIO_RE = re.compile(r"^Load Ratio:\s*([\d.]+)$")


def parse_training_load_assessment(text: str) -> list[ParsedDailyValue]:
    """Each daily block: date line, then Comment/Short-Term Load/Long-Term Load/Load Ratio lines."""
    values: list[ParsedDailyValue] = []
    current_date: str | None = None
    current: dict = {}

    def _flush() -> None:
        if current_date is None:
            return
        values.append(
            ParsedDailyValue(
                date=current_date,
                value=current.get("load_ratio"),
                extra={
                    k: v
                    for k, v in {
                        "comment": current.get("comment"),
                        "short_term_load": current.get("short_term_load"),
                        "long_term_load": current.get("long_term_load"),
                    }.items()
                    if v is not None
                },
            )
        )

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        date_match = _LOAD_DATE_RE.match(line)
        if date_match:
            _flush()
            current_date = date_match.group(1)
            current = {}
            continue
        if (m := _LOAD_COMMENT_RE.match(line)):
            current["comment"] = m.group(1)
        elif (m := _LOAD_SHORT_RE.match(line)):
            current["short_term_load"] = float(m.group(1))
        elif (m := _LOAD_LONG_RE.match(line)):
            current["long_term_load"] = float(m.group(1))
        elif (m := _LOAD_RATIO_RE.match(line)):
            current["load_ratio"] = float(m.group(1))
    _flush()
    return values


_RECOVERY_PCT_RE = re.compile(r"Recovery:\s*(\d+)%")
_RECOVERY_LEVEL_RE = re.compile(r"Level:\s*(.+)")
_RECOVERY_ETA_RE = re.compile(r"Estimated Full Recovery:\s*(.+)")


def parse_recovery_status(text: str) -> dict:
    data: dict = {}
    if (m := _RECOVERY_PCT_RE.search(text)):
        data["recovery_pct"] = int(m.group(1))
    if (m := _RECOVERY_LEVEL_RE.search(text)):
        data["level"] = m.group(1).strip()
    if (m := _RECOVERY_ETA_RE.search(text)):
        data["estimated_full_recovery"] = m.group(1).strip()
    return data


_FITNESS_LINE_RE = re.compile(r"^([\w\s/]+?):\s*(.+)$")
_FITNESS_KEY_MAP = {
    "VO2max": "vo2max",
    "Running Level": "running_level",
    "Threshold Pace": "threshold_pace",
    "5 km Prediction": "prediction_5k",
    "10 km Prediction": "prediction_10k",
    "Half Marathon Prediction": "prediction_half_marathon",
    "Marathon Prediction": "prediction_marathon",
}


def parse_fitness_assessment_overview(text: str) -> dict:
    data: dict = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = _FITNESS_LINE_RE.match(line)
        if not match:
            continue
        label, value = match.group(1).strip(), match.group(2).strip()
        key = _FITNESS_KEY_MAP.get(label)
        if key:
            data[key] = value
    return data

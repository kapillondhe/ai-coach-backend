from pydantic_ai import ModelRetry


def _format_pace(seconds_per_100m: float) -> str:
    """Render seconds-per-100m as 'M:SS per 100m' for a human-readable pace string."""
    minutes, seconds = divmod(round(seconds_per_100m), 60)
    return f"{minutes}:{seconds:02d} per 100m"


_HR_ZONES: list[tuple[int, str, float, float | None]] = [
    (1, "Recovery", 50, 60),
    (2, "Aerobic base / easy", 60, 70),
    (3, "Moderate / tempo-adjacent", 70, 80),
    (4, "Threshold / tempo", 80, 90),
    (5, "VO2max / interval", 90, 100),
]

_POWER_ZONES: list[tuple[int, str, float, float | None]] = [
    (1, "Active recovery", 0, 55),
    (2, "Endurance", 55, 75),
    (3, "Tempo", 75, 90),
    (4, "Threshold", 90, 105),
    (5, "VO2max", 106, 120),
    (6, "Anaerobic capacity", 121, 150),
    (7, "Neuromuscular power", 150, None),
]

_SWIM_ZONES: list[tuple[int, str, float | None, float | None]] = [
    (1, "Recovery", 15, None),
    (2, "Aerobic base", 8, 15),
    (3, "Aerobic threshold", 3, 7),
    (4, "CSS / threshold", -2, 2),
    (5, "VO2max", -8, -3),
    (6, "Sprint / neuromuscular", None, None),
]


async def calculate_heart_rate_zones(max_hr_bpm: float) -> dict:
    """Compute personalized heart-rate training zones (5-zone model) from max heart rate.

    Scales the standard endurance-training heart-rate zone model (percent of max heart
    rate) to a specific athlete's max HR. Use this whenever someone gives (or has
    recorded) a max heart rate and asks what their training zones are, wants to know
    what heart rate an "easy run" or "tempo effort" corresponds to for them, or needs
    zone targets to structure a running/cycling workout by heart rate. A measured max
    HR from a hard effort or race is more reliable than the age-based 220-minus-age
    estimate; if the user only has an estimate, use it but the result carries the same
    uncertainty as the estimate (+/- 10-12 bpm is common error for age-based formulas).

    Args:
        max_hr_bpm: Maximum heart rate in beats per minute. Must be positive.

    Returns:
        A dict echoing max_hr_bpm, plus zones: a list of 5 zone dicts, each with
        zone (1-5), name, pct_range (e.g. "60-70%"), bpm_min, and bpm_max (None for
        an open-ended top zone).
    """
    if max_hr_bpm <= 0:
        raise ModelRetry("max_hr_bpm must be positive")

    zones = []
    for number, name, low_pct, high_pct in _HR_ZONES:
        zones.append(
            {
                "zone": number,
                "name": name,
                "pct_range": f"{low_pct:g}-{high_pct:g}%" if high_pct is not None else f"{low_pct:g}%+",
                "bpm_min": round(max_hr_bpm * low_pct / 100),
                "bpm_max": round(max_hr_bpm * high_pct / 100) if high_pct is not None else None,
            }
        )
    return {"max_hr_bpm": max_hr_bpm, "zones": zones}


async def calculate_power_zones(ftp_watts: float) -> dict:
    """Compute personalized cycling power training zones (7-zone model) from FTP.

    Scales the standard cycling power-zone model (percent of Functional Threshold
    Power) to a specific rider's FTP. Use this whenever someone gives (or has
    recorded) an FTP and asks what their power zones are, wants to know what wattage
    an "endurance ride" or "threshold interval" corresponds to for them, or needs
    zone targets to structure a cycling workout by power. FTP should be a recent
    (within 6-8 weeks) 20-minute-test or ramp-test estimate, or a true 60-minute
    effort — a stale FTP produces zones that under- or overshoot current fitness.

    Args:
        ftp_watts: Functional Threshold Power in watts. Must be positive.

    Returns:
        A dict echoing ftp_watts, plus zones: a list of 7 zone dicts, each with
        zone (1-7), name, pct_range (e.g. "55-75%"), watts_min, and watts_max (None
        for an open-ended top zone).
    """
    if ftp_watts <= 0:
        raise ModelRetry("ftp_watts must be positive")

    zones = []
    for number, name, low_pct, high_pct in _POWER_ZONES:
        zones.append(
            {
                "zone": number,
                "name": name,
                "pct_range": f"{low_pct:g}-{high_pct:g}%" if high_pct is not None else f"{low_pct:g}%+",
                "watts_min": round(ftp_watts * low_pct / 100),
                "watts_max": round(ftp_watts * high_pct / 100) if high_pct is not None else None,
            }
        )
    return {"ftp_watts": ftp_watts, "zones": zones}


async def calculate_swim_pace_zones(css_pace_sec_per_100m: float) -> dict:
    """Compute personalized swim pace training zones (6-zone model) from Critical Swim Speed.

    Scales the standard swim pace-zone model (seconds per 100m relative to Critical
    Swim Speed, CSS) to a specific swimmer's CSS. Use this whenever someone gives (or
    has recorded) a CSS pace and asks what their swim training zones are, wants to
    know what pace an "aerobic base set" or "CSS interval" corresponds to for them, or
    needs pace targets to structure a swim workout. CSS is derived from a 400m and
    200m time trial: CSS pace per 100m = (400m time - 200m time) / 2, in seconds.
    Should be retested every 6-8 weeks during a structured block.

    Args:
        css_pace_sec_per_100m: Critical Swim Speed pace, in seconds per 100m. Must be
            positive (e.g. 90 for a 1:30/100m CSS pace).

    Returns:
        A dict echoing css_pace_sec_per_100m (and a human-readable css_pace string),
        plus zones: a list of 6 zone dicts, each with zone (1-6), name, offset_range
        (e.g. "CSS +8 to +15 sec/100m", or "maximal effort, no pace target" for the
        sprint zone), pace_sec_min, pace_sec_max, and human-readable pace_min/pace_max
        strings (None for the open-ended recovery zone's slow bound, or entirely None
        for the sprint zone with no pace target).
    """
    if css_pace_sec_per_100m <= 0:
        raise ModelRetry("css_pace_sec_per_100m must be positive")

    zones = []
    for number, name, offset_min, offset_max in _SWIM_ZONES:
        if offset_min is None and offset_max is None:
            zones.append(
                {
                    "zone": number,
                    "name": name,
                    "offset_range": "maximal effort, no pace target",
                    "pace_sec_min": None,
                    "pace_sec_max": None,
                    "pace_min": None,
                    "pace_max": None,
                }
            )
            continue

        bounds = [o for o in (offset_min, offset_max) if o is not None]
        fast_offset = min(bounds)
        slow_offset = max(bounds) if offset_max is not None and offset_min is not None else None

        pace_sec_min = css_pace_sec_per_100m + fast_offset
        pace_sec_max = css_pace_sec_per_100m + slow_offset if slow_offset is not None else None

        def _fmt_offset(o: float) -> str:
            return f"+{o:g}" if o >= 0 else f"{o:g}"

        assert offset_min is not None  # only the all-out sprint zone has no bounds, handled above
        if offset_max is None:
            offset_range = f"CSS {_fmt_offset(offset_min)}+ sec/100m"
        elif offset_min == offset_max:
            offset_range = f"CSS {_fmt_offset(offset_min)} sec/100m"
        else:
            offset_range = f"CSS {_fmt_offset(offset_min)} to {_fmt_offset(offset_max)} sec/100m"

        zones.append(
            {
                "zone": number,
                "name": name,
                "offset_range": offset_range,
                "pace_sec_min": round(pace_sec_min, 1),
                "pace_sec_max": round(pace_sec_max, 1) if pace_sec_max is not None else None,
                "pace_min": _format_pace(pace_sec_min),
                "pace_max": _format_pace(pace_sec_max) if pace_sec_max is not None else None,
            }
        )

    return {
        "css_pace_sec_per_100m": css_pace_sec_per_100m,
        "css_pace": _format_pace(css_pace_sec_per_100m),
        "zones": zones,
    }

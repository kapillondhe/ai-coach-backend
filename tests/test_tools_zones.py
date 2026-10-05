import pytest
from pydantic_ai import ModelRetry

from app.tools.zones import (
    calculate_heart_rate_zones,
    calculate_power_zones,
    calculate_swim_pace_zones,
)


async def test_calculate_heart_rate_zones_basic():
    result = await calculate_heart_rate_zones(190)
    assert result["max_hr_bpm"] == 190
    assert len(result["zones"]) == 5
    zone2 = result["zones"][1]
    assert zone2["zone"] == 2
    assert zone2["bpm_min"] == round(190 * 0.6)
    assert zone2["bpm_max"] == round(190 * 0.7)
    assert result["zones"][-1]["bpm_max"] == 190


async def test_calculate_heart_rate_zones_rejects_non_positive():
    with pytest.raises(ModelRetry):
        await calculate_heart_rate_zones(0)


async def test_calculate_power_zones_basic():
    result = await calculate_power_zones(250)
    assert result["ftp_watts"] == 250
    assert len(result["zones"]) == 7
    threshold = result["zones"][3]
    assert threshold["zone"] == 4
    assert threshold["watts_min"] == round(250 * 0.90)
    assert threshold["watts_max"] == round(250 * 1.05)
    # zone 7 (neuromuscular power) is open-ended
    assert result["zones"][-1]["watts_max"] is None
    assert result["zones"][-1]["watts_min"] == round(250 * 1.50)


async def test_calculate_power_zones_rejects_non_positive():
    with pytest.raises(ModelRetry):
        await calculate_power_zones(-10)


async def test_calculate_swim_pace_zones_basic():
    result = await calculate_swim_pace_zones(90)
    assert result["css_pace_sec_per_100m"] == 90
    assert result["css_pace"] == "1:30 per 100m"
    assert len(result["zones"]) == 6

    css_zone = result["zones"][3]
    assert css_zone["zone"] == 4
    assert css_zone["pace_sec_min"] == 88
    assert css_zone["pace_sec_max"] == 92

    recovery_zone = result["zones"][0]
    assert recovery_zone["pace_sec_min"] == 105
    assert recovery_zone["pace_sec_max"] is None

    sprint_zone = result["zones"][-1]
    assert sprint_zone["pace_sec_min"] is None
    assert sprint_zone["pace_sec_max"] is None
    assert sprint_zone["pace_min"] is None


async def test_calculate_swim_pace_zones_faster_offset_is_smaller_seconds():
    result = await calculate_swim_pace_zones(90)
    vo2max_zone = result["zones"][4]
    assert vo2max_zone["zone"] == 5
    # VO2max zone is faster than CSS: offsets are -8 to -3, so pace_sec_min (fastest) < pace_sec_max
    assert vo2max_zone["pace_sec_min"] == 82
    assert vo2max_zone["pace_sec_max"] == 87


async def test_calculate_swim_pace_zones_rejects_non_positive():
    with pytest.raises(ModelRetry):
        await calculate_swim_pace_zones(0)

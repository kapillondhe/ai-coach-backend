import pytest
from pydantic_ai import ModelRetry

from app.tools.nutrition import calculate_protein_intake


async def test_calculate_protein_intake_general_fitness():
    result = await calculate_protein_intake(70, goal="general_fitness")
    assert result == {
        "weight_kg": 70,
        "goal": "general_fitness",
        "protein_g_min": 84.0,
        "protein_g_max": 112.0,
    }


async def test_calculate_protein_intake_defaults_to_general_fitness():
    result = await calculate_protein_intake(70)
    assert result["goal"] == "general_fitness"


async def test_calculate_protein_intake_muscle_gain_is_higher_than_sedentary():
    sedentary = await calculate_protein_intake(80, goal="sedentary")
    muscle_gain = await calculate_protein_intake(80, goal="muscle_gain")
    assert muscle_gain["protein_g_min"] > sedentary["protein_g_max"]


async def test_calculate_protein_intake_rejects_non_positive_weight():
    # ModelRetry, not ValueError: the model sees the message and can correct its args.
    with pytest.raises(ModelRetry):
        await calculate_protein_intake(0)

from typing import Literal

from pydantic_ai import ModelRetry

ProteinGoal = Literal["sedentary", "general_fitness", "muscle_gain", "fat_loss", "endurance"]

_GRAMS_PER_KG: dict[ProteinGoal, tuple[float, float]] = {
    "sedentary": (0.8, 1.0),
    "general_fitness": (1.2, 1.6),
    "muscle_gain": (1.6, 2.2),
    "fat_loss": (1.8, 2.4),
    "endurance": (1.2, 1.6),
}


async def calculate_protein_intake(
    weight_kg: float,
    goal: ProteinGoal = "general_fitness",
) -> dict:
    """Recommend a daily dietary protein range, in grams, for a person.

    Scales standard sports-nutrition intake guidelines (grams of protein per
    kilogram of body weight per day) by the person's body weight and why they
    need the guidance. Use this whenever someone asks how much protein they
    should eat, whether their current intake is enough, or wants a nutrition
    target to go with a training plan.

    Args:
        weight_kg: Body weight in kilograms. Must be positive.
        goal: Why this person needs protein guidance, which sets the target range:
            - "sedentary": little to no regular exercise; general health upkeep.
            - "general_fitness": recreational exercise a few times a week.
            - "muscle_gain": actively training to build muscle (hypertrophy/strength).
            - "fat_loss": eating in a calorie deficit; extra protein helps preserve
              lean muscle mass while losing fat.
            - "endurance": regular endurance training, e.g. running or cycling.

    Returns:
        A dict echoing weight_kg and goal, plus protein_g_min and protein_g_max —
        the recommended daily protein intake range in grams.
    """
    if weight_kg <= 0:
        raise ModelRetry("weight_kg must be positive")

    low_factor, high_factor = _GRAMS_PER_KG[goal]
    return {
        "weight_kg": weight_kg,
        "goal": goal,
        "protein_g_min": round(weight_kg * low_factor, 1),
        "protein_g_max": round(weight_kg * high_factor, 1),
    }

"""Measure the playbook router's accuracy against labeled messages (live Jev calls).

Each case lists the playbooks a good router must pick (`required`) and the ones it may
also pick without being wrong (`allowed`). A pick outside required+allowed is a false
positive; a missing required one is a miss. Re-run after editing a playbook's
`description`, adding a playbook, or changing the router's instructions or model —
unit tests mock Jev and can't catch a routing regression.

Usage (from `ai-coach-backend/`, venv active, OPENROUTER_API_KEY in .env):

    python -m evals.playbook_routing
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field

from app.services import playbooks


@dataclass(frozen=True)
class RoutingCase:
    message: str
    required: set[str]
    allowed: set[str] = field(default_factory=set)


CASES = [
    # intensity-and-zones
    RoutingCase("Is my easy run too fast if my heart rate is 160?", {"intensity-and-zones"}),
    RoutingCase("How do I test my FTP?", {"intensity-and-zones"}),
    RoutingCase("What is zone 2 and why does everyone talk about it?", {"intensity-and-zones"}),
    # training-week
    RoutingCase("How many hours a week should I train for my first 70.3?", {"training-week"}, {"race-execution"}),
    RoutingCase(
        "Can you give me a sample week for Olympic distance? I have 7 hours.",
        {"training-week"},
        {"intensity-and-zones"},
    ),
    RoutingCase("How often should I do brick workouts?", {"training-week"}),
    # adjusting-the-plan
    RoutingCase(
        "I've had a cold for 4 days, can I still do my long ride?",
        {"adjusting-the-plan"},
        {"recovery-and-readiness"},
    ),
    RoutingCase("Work is crazy this week, I can only train 3 hours. What should I keep?", {"adjusting-the-plan"}),
    RoutingCase("I missed two weeks on holiday, how do I get back into it?", {"adjusting-the-plan"}, {"training-week"}),
    # race-execution
    RoutingCase("How long should my taper be for an Ironman?", {"race-execution"}),
    RoutingCase(
        "What power should I hold on the bike in a 70.3 if my FTP is 240?",
        {"race-execution"},
        {"intensity-and-zones"},
    ),
    RoutingCase("My legs feel awful running off the bike, any tips for T2?", {"race-execution"}, {"training-week"}),
    # fueling-and-hydration
    RoutingCase("How many carbs per hour should I take on a 4 hour ride?", {"fueling-and-hydration"}),
    RoutingCase("Should I carb load before an Olympic triathlon?", {"fueling-and-hydration"}, {"race-execution"}),
    RoutingCase("I get stomach cramps from gels during long runs", {"fueling-and-hydration"}),
    # recovery-and-readiness
    RoutingCase("My HRV has been low all week, should I rest?", {"recovery-and-readiness"}),
    RoutingCase("I feel flat and tired on every session lately", {"recovery-and-readiness"}, {"adjusting-the-plan"}),
    RoutingCase("How much sleep do I need while training for an Ironman?", {"recovery-and-readiness"}),
    # pain-and-injury
    RoutingCase("My Achilles is stiff every morning, should I keep running?", {"pain-and-injury"}),
    RoutingCase(
        "My knee hurts on the outside after about 10k",
        {"pain-and-injury"},
        {"strength-and-mobility"},
    ),
    RoutingCase(
        "I get lower back pain on long rides on my tri bike",
        {"pain-and-injury"},
        {"strength-and-mobility"},
    ),
    # strength-and-mobility
    RoutingCase("Should triathletes lift weights?", {"strength-and-mobility"}),
    RoutingCase("What strength exercises help with running economy?", {"strength-and-mobility"}),
    # swim-and-open-water
    RoutingCase("How do I stop panicking at a mass swim start?", {"swim-and-open-water"}, {"race-execution"}),
    RoutingCase("What drills fix a dropped elbow in freestyle?", {"swim-and-open-water"}),
    RoutingCase("Is a wetsuit worth it for a sprint triathlon?", {"swim-and-open-water"}, {"race-execution"}),
    # two topics
    RoutingCase(
        "My calf is tight and my marathon is in 3 weeks, should I still taper normally?",
        {"pain-and-injury", "race-execution"},
        {"adjusting-the-plan"},
    ),
    RoutingCase(
        "What should I eat and how should I pace the run in a 70.3?",
        {"fueling-and-hydration", "race-execution"},
    ),
    # none
    RoutingCase("Hi!", set()),
    RoutingCase("Thanks, that's really helpful", set()),
    RoutingCase("What can you help me with?", set(), {"training-week"}),
]


async def _route(case: RoutingCase) -> tuple[RoutingCase, list[str], float]:
    start = time.perf_counter()
    picked = await playbooks.select_playbooks(case.message)
    return case, picked, time.perf_counter() - start


async def main() -> int:
    results = await asyncio.gather(*(_route(case) for case in CASES))

    exact = misses = false_positives = 0
    for case, picked, seconds in results:
        got = set(picked)
        missing = case.required - got
        extra = got - case.required - case.allowed
        ok = not missing and not extra
        exact += ok
        misses += bool(missing)
        false_positives += bool(extra)
        mark = "ok  " if ok else "FAIL"
        detail = "".join(
            [f"  missing={sorted(missing)}" if missing else "", f"  extra={sorted(extra)}" if extra else ""]
        )
        print(f"{mark} {seconds:4.1f}s  {case.message[:70]:<70} -> {picked}{detail}")

    latencies = sorted(seconds for _, _, seconds in results)
    print(
        f"\n{exact}/{len(CASES)} correct, {misses} with a missed playbook, {false_positives} with an "
        f"unwanted one. Latency median {latencies[len(latencies) // 2]:.2f}s, max {latencies[-1]:.2f}s "
        "(all cases run concurrently, so this is under load)."
    )
    return 0 if exact == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

"""The hazards detection/ looks for, and how to map model labels back to them.

Keys match the planned ``HazardType`` values in the shared contract
(``spill``, ``unattended_burner``, ``raw_meat_contact``, ``knife_edge``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Hazard:
    key: str
    label: str  # short human-readable name for the overlay
    description: str  # what the model is told to look for
    color: tuple[int, int, int]  # BGR, for OpenCV
    keywords: tuple[str, ...]  # word prefixes, for when the model paraphrases the label


HAZARDS: dict[str, Hazard] = {
    h.key: h
    for h in (
        Hazard(
            key="knife_edge",
            label="Sharp tool",
            description=(
                "a knife, cleaver, peeler, mandoline, or other sharp tool left unattended: lying with the blade "
                "exposed, hanging over a counter edge, left in a sink or on a cutting board with no one using it"
            ),
            color=(0, 165, 255),
            keywords=("knife", "blade", "cleaver", "sharp", "scissor", "peeler", "mandoline", "edge"),
        ),
        Hazard(
            key="spill",
            label="Spill",
            description="liquid, grease, oil, food, or ice spilled on the floor or a work surface",
            color=(255, 200, 0),
            keywords=("spill", "liquid", "puddle", "water", "grease", "oil", "wet", "slip"),
        ),
        Hazard(
            key="unattended_burner",
            label="Unattended burner",
            description=(
                "a lit burner, open flame, fire, or smoking pan on a stove with no person attending it"
            ),
            color=(0, 0, 255),
            keywords=("burner", "flame", "fire", "stove", "smok", "pan", "pot", "heat"),
        ),
        Hazard(
            key="raw_meat_contact",
            label="Raw food contamination",
            description=(
                "raw meat, poultry, or fish touching, dripping onto, or sharing a cutting board or utensil with "
                "cooked or ready-to-eat food or produce"
            ),
            color=(255, 0, 255),
            keywords=("raw", "meat", "chicken", "poultry", "fish", "beef", "pork", "contamin", "cross"),
        ),
    )
}


def match_hazard(mention: str | None) -> str | None:
    """Return the hazard key a model label refers to, or None if it matches none."""
    if not mention:
        return None
    text = mention.strip().lower()
    normalized = text.replace(" ", "_").replace("-", "_")
    if normalized in HAZARDS:
        return normalized
    # Most specific first: "raw chicken in a pan" is contamination, not a burner.
    for key in _MATCH_ORDER:
        if any(re.search(rf"\b{word}", text) for word in HAZARDS[key].keywords):
            return key
    return None


_MATCH_ORDER = ("raw_meat_contact", "knife_edge", "spill", "unattended_burner")

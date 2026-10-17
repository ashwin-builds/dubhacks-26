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
            # Unattended = set down with nobody using it. A knife in use is normal work.
            description=(
                "a knife, cleaver, peeler, mandoline, or other sharp tool that has been left unattended: set down "
                "with no one holding it and no one standing at it working (for example left on a counter, cutting "
                "board, shelf, stove, floor, or in a sink while the cook has walked away). "
                "NOT a hazard: a knife in someone's hand, including while they are cutting, chopping, slicing, or "
                "carrying it; a knife a cook has just put down on their own cutting board while they keep working "
                "right there; knives stored in a knife block, sheath, magnetic strip, or drawer. "
                "Box only the sharp tool, not the person"
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
            # Unattended = on with no cook at the stove, with or without a pan on it.
            description=(
                "a stove burner, grill, or griddle that is on (visible flame, glowing red element, steam, or "
                "sizzling food) with no worker standing at the stove tending it. A pan or pot on an unattended lit "
                "burner is still a hazard. "
                "NOT a hazard: a burner that is on while someone is at the stove cooking (stirring, flipping, "
                "plating, or standing in front of it watching the food); a burner that is off. "
                "Box only the burner, not the person"
            ),
            color=(0, 0, 255),
            keywords=("burner", "flame", "fire", "stove", "heat"),
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

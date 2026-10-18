"""Ignore spots a manager has marked as false positives.

When the manager marks an incident as a false positive, the boxes it was last seen
in become ignore zones for that hazard. Later detections of the same hazard
that mostly overlap a zone are dropped before they reach the incident detector,
so the same misread object doesn't open a new incident and text the manager
again. Other hazards in the same spot, and the same hazard elsewhere in the
frame, are unaffected.

A zone ends when either:

* it is ``max_age_seconds`` old (a mistaken "no" can't blind a spot forever), or
* nothing has matched it for ``idle_seconds`` (the misread object is gone).

Every dropped detection is logged and counted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from detection.incidents import Box, Danger

log = logging.getLogger(__name__)

# TODO(zach): placeholders -- how long a rejection lasts.
MAX_AGE_SECONDS = 30 * 60.0
IDLE_SECONDS = 60.0

# A detection matches a zone when this much of the smaller box lies inside the other. Using the smaller box,
# not intersection-over-union, tolerates the model drawing a tighter or looser box around the same object.
MIN_OVERLAP = 0.5


@dataclass
class IgnoreZone:
    danger: Danger
    box: Box
    created_at: datetime
    last_matched: datetime
    incident_id: str
    suppressed: int = 0


def overlap(a: Box, b: Box) -> float:
    """Intersection area as a fraction of the smaller box's area."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return width * height / smaller if smaller > 0 else 0.0


class Suppressor:
    def __init__(
        self,
        max_age_seconds: float = MAX_AGE_SECONDS,
        idle_seconds: float = IDLE_SECONDS,
        min_overlap: float = MIN_OVERLAP,
    ):
        self.max_age_seconds = max_age_seconds
        self.idle_seconds = idle_seconds
        self.min_overlap = min_overlap
        self.zones: list[IgnoreZone] = []
        self.suppressed = 0  # detections dropped over the whole stream

    def add(self, danger: Danger, boxes: tuple[Box, ...], now: datetime, incident_id: str) -> None:
        for box in boxes:
            self.zones.append(IgnoreZone(danger, box, now, now, incident_id))
            log.info("ignoring %s at %s (false positive %s)", danger.value, box, incident_id)

    def is_suppressed(self, danger: Danger, box: Box, now: datetime) -> bool:
        """True if this detection falls in an ignore zone for its hazard. Call in frame order."""
        self._expire(now)
        matched = False
        for zone in self.zones:
            if zone.danger == danger and overlap(zone.box, box) >= self.min_overlap:
                zone.last_matched = now
                zone.suppressed += 1
                matched = True
        if matched:
            self.suppressed += 1
            log.info("suppressed %s at %s: inside a false-positive zone", danger.value, box)
        return matched

    def _expire(self, now: datetime) -> None:
        kept = []
        for zone in self.zones:
            age = (now - zone.created_at).total_seconds()
            idle = (now - zone.last_matched).total_seconds()
            if age >= self.max_age_seconds or idle >= self.idle_seconds:
                log.info(
                    "no longer ignoring %s at %s (%d detection(s) suppressed)",
                    zone.danger.value,
                    zone.box,
                    zone.suppressed,
                )
            else:
                kept.append(zone)
        self.zones = kept

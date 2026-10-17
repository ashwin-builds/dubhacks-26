from datetime import datetime, timedelta, timezone

from detection.incidents import Danger
from detection.suppression import Suppressor, overlap

T0 = datetime(2026, 10, 17, 18, 0, tzinfo=timezone.utc)
KNIFE = (100, 100, 200, 200)


def at(seconds):
    return T0 + timedelta(seconds=seconds)


def test_overlap_uses_the_smaller_box():
    assert overlap(KNIFE, KNIFE) == 1.0
    assert overlap(KNIFE, (120, 120, 180, 180)) == 1.0  # tighter box around the same object
    assert overlap(KNIFE, (150, 100, 250, 200)) == 0.5
    assert overlap(KNIFE, (300, 300, 400, 400)) == 0.0
    assert overlap(KNIFE, (200, 100, 300, 200)) == 0.0  # touching edges only


def test_same_hazard_same_spot_is_suppressed():
    s = Suppressor()
    s.add(Danger.SHARPS, (KNIFE,), at(0), "inc1")
    assert s.is_suppressed(Danger.SHARPS, (110, 105, 205, 195), at(1))
    assert s.suppressed == 1


def test_other_hazards_and_other_spots_are_not():
    s = Suppressor()
    s.add(Danger.SHARPS, (KNIFE,), at(0), "inc1")
    assert not s.is_suppressed(Danger.SPILLS, KNIFE, at(1))
    assert not s.is_suppressed(Danger.SHARPS, (600, 600, 700, 700), at(1))
    assert s.suppressed == 0


def test_zone_expires_when_nothing_matches_for_a_while():
    s = Suppressor(idle_seconds=60)
    s.add(Danger.SHARPS, (KNIFE,), at(0), "inc1")
    assert s.is_suppressed(Danger.SHARPS, KNIFE, at(50))
    assert s.is_suppressed(Danger.SHARPS, KNIFE, at(100))  # matched at 50, so still alive
    assert not s.is_suppressed(Danger.SHARPS, KNIFE, at(161))  # gone for over 60 s
    assert s.zones == []


def test_zone_expires_after_max_age_even_if_still_matching():
    s = Suppressor(max_age_seconds=300, idle_seconds=60)
    s.add(Danger.SHARPS, (KNIFE,), at(0), "inc1")
    for t in range(10, 300, 10):
        assert s.is_suppressed(Danger.SHARPS, KNIFE, at(t))
    assert not s.is_suppressed(Danger.SHARPS, KNIFE, at(300))


def test_all_boxes_of_the_rejected_incident_become_zones():
    s = Suppressor()
    s.add(Danger.SHARPS, (KNIFE, (600, 600, 700, 700)), at(0), "inc1")
    assert s.is_suppressed(Danger.SHARPS, (600, 600, 700, 700), at(1))
    assert len(s.zones) == 2

from types import SimpleNamespace

import numpy as np
import pytest

from detection.hazards import match_hazard
from detection.isaac_client import (
    DEFAULT_MODEL,
    Detection,
    IsaacClient,
    build_system_prompt,
    encode_frame,
    split_confidence,
)
from detection.overlay import draw_detections, draw_status


def _box(mention, x1, y1, x2, y2):
    return SimpleNamespace(
        mention=mention,
        top_left=SimpleNamespace(x=x1, y=y1),
        bottom_right=SimpleNamespace(x=x2, y=y2),
    )


def _fake_perceive(boxes, calls=None):
    def perceive(*nodes, **kwargs):
        if calls is not None:
            calls.append((nodes, kwargs))
        return SimpleNamespace(boxes=boxes, text="")

    return perceive


@pytest.mark.parametrize(
    ("mention", "expected"),
    [
        ("spill", "spill"),
        ("knife_edge", "knife_edge"),
        ("Unattended Burner", "unattended_burner"),
        ("raw-meat-contact", "raw_meat_contact"),
        ("water puddle on floor", "spill"),
        ("chef's knife near edge", "knife_edge"),
        ("raw chicken in a pan", "raw_meat_contact"),
        ("open flame", "unattended_burner"),
        ("drawer", None),
        ("person", None),
        ("", None),
        (None, None),
    ],
)
def test_match_hazard(mention, expected):
    assert match_hazard(mention) == expected


def test_detect_maps_boxes_and_drops_unknown_labels():
    boxes = [_box("spill", 100, 700, 300, 900), _box("person", 0, 0, 500, 1000), _box("knife", 600, 400, 500, 300)]
    calls = []
    client = IsaacClient(perceive_fn=_fake_perceive(boxes, calls))

    result = client.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    assert [d.hazard for d in result.detections] == ["spill", "knife_edge"]
    assert result.detections[1].box == (500, 300, 600, 400)  # corners normalised
    assert result.unmatched == ["person"]
    _, kwargs = calls[0]
    assert kwargs["expects"] == "box"


def test_detect_ignores_hazards_not_requested():
    client = IsaacClient(hazards=["spill"], perceive_fn=_fake_perceive([_box("knife_edge", 0, 0, 10, 10)]))
    result = client.detect(np.zeros((10, 10, 3), dtype=np.uint8))
    assert result.detections == []
    assert result.unmatched == ["knife_edge"]


def test_detect_handles_no_boxes():
    client = IsaacClient(perceive_fn=_fake_perceive(None))
    assert client.detect(np.zeros((10, 10, 3), dtype=np.uint8)).detections == []


def test_unknown_hazard_rejected():
    with pytest.raises(ValueError):
        IsaacClient(hazards=["glitter"], perceive_fn=_fake_perceive([]))


def test_system_prompt_lists_only_requested_hazards():
    prompt = build_system_prompt(["spill", "knife_edge"])
    assert "spill" in prompt and "knife_edge" in prompt
    assert "unattended_burner" not in prompt


def test_pixel_box_scales_from_grid():
    assert Detection("spill", "spill", (0, 500, 1000, 1000)).pixel_box(640, 480) == (0, 240, 640, 480)


def test_encode_frame_downscales_wide_frames():
    import cv2

    data = encode_frame(np.zeros((1080, 1920, 3), dtype=np.uint8))
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[1] == 1024


def test_overlay_draws_without_mutating_input():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = draw_status(frame, "HAZARD", alert=True)
    out = draw_detections(out, [Detection("spill", "spill", (100, 100, 400, 400))])
    assert frame.sum() == 0
    assert out.sum() > 0


@pytest.mark.parametrize(
    ("mention", "expected"),
    [
        ("spill 0.8", ("spill", 0.8)),
        ("knife_edge: 0.65", ("knife_edge", 0.65)),
        ("unattended_burner (90%)", ("unattended_burner", 0.9)),
        ("raw_meat_contact 85", ("raw_meat_contact", 0.85)),
        ("spill 1", ("spill", 1.0)),
        ("spill", ("spill", None)),
        ("spill 250", ("spill 250", None)),
        ("0.9", ("0.9", None)),
    ],
)
def test_split_confidence(mention, expected):
    assert split_confidence(mention) == expected


def test_detect_reads_confidence_from_label():
    boxes = [_box("spill 0.72", 0, 0, 10, 10), _box("knife_edge", 0, 0, 10, 10)]
    result = IsaacClient(perceive_fn=_fake_perceive(boxes)).detect(np.zeros((10, 10, 3), dtype=np.uint8))
    assert [(d.hazard, d.confidence) for d in result.detections] == [("spill", 0.72), ("knife_edge", None)]


def test_model_defaults_to_fast_isaac(monkeypatch):
    monkeypatch.delenv("PERCEPTRON_MODEL", raising=False)
    calls = []
    client = IsaacClient(perceive_fn=_fake_perceive([], calls))
    client.detect(np.zeros((10, 10, 3), dtype=np.uint8))
    assert client.model == DEFAULT_MODEL == "isaac-0.3-fast"
    assert calls[0][1]["model"] == "isaac-0.3-fast"


def test_model_override_order(monkeypatch):
    monkeypatch.setenv("PERCEPTRON_MODEL", "perceptron-mk1.5")
    assert IsaacClient(perceive_fn=_fake_perceive([])).model == "perceptron-mk1.5"
    assert IsaacClient(model="isaac-0.1", perceive_fn=_fake_perceive([])).model == "isaac-0.1"


def test_prompt_states_unattended_rules():
    prompt = build_system_prompt(["knife_edge", "unattended_burner"])
    assert "NOT on a cutting board" in prompt and "no one is holding" in prompt
    assert "no pan or pot on it" in prompt


def test_overlay_draws_confidence():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    without = draw_detections(frame, [Detection("spill", "spill", (100, 100, 900, 900))])
    with_conf = draw_detections(frame, [Detection("spill", "spill 0.8", (100, 100, 900, 900), confidence=0.8)])
    assert not np.array_equal(without, with_conf)

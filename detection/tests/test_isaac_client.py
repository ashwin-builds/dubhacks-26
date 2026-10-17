from types import SimpleNamespace

import numpy as np
import pytest

from detection.hazards import match_hazard
from detection.isaac_client import (
    FALLBACK_MODEL,
    Detection,
    IsaacClient,
    build_system_prompt,
    encode_frame,
    pick_model,
)
from detection.overlay import draw_detections, draw_status


@pytest.fixture(autouse=True)
def _no_model_listing(monkeypatch):
    monkeypatch.setattr("detection.isaac_client.list_models", lambda: ["isaac-0.1"])


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
    ("available", "expected"),
    [
        (["perceptron-mk1.5", "isaac-0.2-2b-preview", "isaac-0.1"], "isaac-0.2-2b-preview"),
        (["perceptron-mk1.5", "perceptron-mk1"], "perceptron-mk1"),
        (["perceptron-mk1.5", "isaac-0.4-turbo"], "isaac-0.4-turbo"),
        (["some-new-model"], "some-new-model"),
        ([], FALLBACK_MODEL),
    ],
)
def test_pick_model(available, expected):
    assert pick_model(available) == expected


def test_model_auto_picks_from_available(monkeypatch):
    monkeypatch.delenv("PERCEPTRON_MODEL", raising=False)
    calls = []
    client = IsaacClient(perceive_fn=_fake_perceive([], calls), list_models_fn=lambda: ["perceptron-mk1.5", "isaac-0.1"])
    client.detect(np.zeros((10, 10, 3), dtype=np.uint8))
    assert client.model == "isaac-0.1"
    assert calls[0][1]["model"] == "isaac-0.1"


def test_model_falls_back_when_listing_fails(monkeypatch):
    monkeypatch.delenv("PERCEPTRON_MODEL", raising=False)

    def broken():
        raise RuntimeError("no network")

    assert IsaacClient(perceive_fn=_fake_perceive([]), list_models_fn=broken).model == FALLBACK_MODEL


def test_model_override_order(monkeypatch):
    monkeypatch.setenv("PERCEPTRON_MODEL", "perceptron-mk1.5")
    never = lambda: pytest.fail("should not list models when a model is given")  # noqa: E731
    assert IsaacClient(perceive_fn=_fake_perceive([]), list_models_fn=never).model == "perceptron-mk1.5"
    assert IsaacClient(model="isaac-0.1", perceive_fn=_fake_perceive([]), list_models_fn=never).model == "isaac-0.1"


def test_prompt_states_unattended_rules():
    prompt = build_system_prompt(["knife_edge", "unattended_burner"])
    assert "left unattended" in prompt and "while they are cutting" in prompt
    assert "no worker standing at the stove" in prompt and "while someone is at the stove cooking" in prompt
    assert "normal kitchen work" in prompt

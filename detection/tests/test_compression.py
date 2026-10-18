import cv2
import numpy as np

from detection.compression import compress_frame


def test_compress_frame_shrinks_to_jpeg():
    big = np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    jpeg = compress_frame(big)
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert jpeg[:2] == b"\xff\xd8"  # JPEG magic
    assert image.shape[:2] == (360, 640)
    assert len(jpeg) < big.nbytes / 20


def test_compress_frame_accepts_encoded_bytes():
    ok, png = cv2.imencode(".png", np.zeros((50, 100, 3), np.uint8))
    image = cv2.imdecode(np.frombuffer(compress_frame(png.tobytes()), np.uint8), cv2.IMREAD_COLOR)
    assert image.shape[:2] == (50, 100)  # never upscaled

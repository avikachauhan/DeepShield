"""
DeepShield — utils/preprocess.py
Image loading and preprocessing helpers used by both train.py and app.py.
"""

import numpy as np
from PIL import Image, UnidentifiedImageError

IMG_SIZE = (224, 224)


def load_and_preprocess(source) -> np.ndarray:
    """
    Load an image from a file path or a file-like object (e.g. Streamlit
    UploadedFile) and return a normalised numpy array ready for inference.

    Returns shape: (1, 224, 224, 3), dtype: float32, values in [0, 1].
    Raises ValueError for unreadable or invalid images.
    """
    try:
        img = Image.open(source).convert("RGB")
    except (UnidentifiedImageError, Exception) as exc:
        raise ValueError(f"Cannot open image: {exc}") from exc

    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0   # normalise to [0, 1]
    return np.expand_dims(arr, axis=0)               # add batch dimension


def decode_prediction(raw_score: float, threshold: float = 0.5) -> tuple[str, float]:
    """
    Convert the model's sigmoid output to a human-readable label and
    a confidence percentage.

    Args:
        raw_score: float in [0, 1] — model output (1 = fake, 0 = real).
        threshold: decision boundary (default 0.5).

    Returns:
        (label, confidence_pct)
        label          : "FAKE" or "REAL"
        confidence_pct : float in [0, 100]
    """
    if raw_score >= threshold:
        label = "FAKE"
        confidence = raw_score * 100
    else:
        label = "REAL"
        confidence = (1.0 - raw_score) * 100

    return label, round(confidence, 2)

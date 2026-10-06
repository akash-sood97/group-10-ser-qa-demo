"""Scoring engine for the QA demo.

Takes the bytes of an uploaded call recording and returns the predicted
emotion, the confidence and whether the call is flagged for QA review.
Every step before the embedding reuses ``shared/audio_utils.py``, so the
audio reaches the model exactly as it did in the notebooks.

Pipeline: decode -> mono -> 16 kHz -> trim silence -> windows
-> frozen WavLM embedding (layer 5, mean over time) -> numpy head
-> mean over windows -> flag rule.

Importing this module never downloads anything and never imports torch
or transformers; the encoder is loaded on the first call that needs it.
"""

import io
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SHARED = next(p / "shared" for p in [HERE, *HERE.parents]
              if (p / "shared" / "audio_utils.py").exists())
sys.path.insert(0, str(SHARED))

import audio_utils as U  # noqa: E402

# Model pinned to one revision so results never change under us.
CHECKPOINT = "microsoft/wavlm-base-plus"
REVISION = "4c66d4806a428f2e922ccfa1a962776e232d487b"

# Input limits.
MAX_SECONDS = 60.0
MAX_BYTES = 20 * 1024 * 1024
SILENCE_PEAK = 1e-4
MIN_SPEECH_S = 0.5

# Windowing: speech up to 6 s is scored as one window, longer speech
# is scored in 3.5 s windows with 50% overlap.
SINGLE_WINDOW_MAX = 96000
HOP = 28000


class ScoringError(Exception):
    """An input problem, with a plain-English message for the user."""


# ------------------------------------------------------------------- #
# Assets
# ------------------------------------------------------------------- #

def load_assets(assets_dir):
    """Load the head, its metadata, the flag rule, queue and clip list.

    Parameters
    ----------
    assets_dir : str or Path
        Folder holding ``wavlm_head.npz`` and its companion files.

    Returns
    -------
    dict
        ``head`` (arrays), ``meta``, ``flag`` (flag rule), ``queue`` and
        ``clips`` (DataFrames).
    """
    import json

    a = Path(assets_dir)
    z = np.load(a / "wavlm_head.npz", allow_pickle=False)
    head = {
        "mean": z["mean"].astype(np.float64),
        "scale": z["scale"].astype(np.float64),
        "coef": z["coef"].astype(np.float64),
        "intercept": z["intercept"].astype(np.float64),
        "classes": [str(c) for c in z["classes"]],
        "layer": int(z["layer"]),
    }
    return {
        "head": head,
        "meta": json.loads((a / "head_meta.json").read_text()),
        "flag": json.loads((a / "flag_operating_point.json").read_text()),
        "queue": pd.read_csv(a / "heldout_queue.csv", dtype={"speaker": str}),
        "clips": pd.read_csv(a / "clips" / "manifest.csv",
                             dtype={"speaker": str}),
    }


# ------------------------------------------------------------------- #
# Audio preparation
# ------------------------------------------------------------------- #

def decode(data: bytes):
    """Decode wav, flac, ogg or mp3 bytes to mono float32.

    Returns
    -------
    tuple
        ``(samples, sample_rate_hz)``.
    """
    import soundfile as sf

    if len(data) > MAX_BYTES:
        raise ScoringError(
            f"This file is {len(data) / 1e6:.1f} MB. The limit is "
            f"{MAX_BYTES // (1024 * 1024)} MB.")
    try:
        x, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ScoringError(
            "This file could not be read as audio. Please upload a "
            "wav, flac, ogg or mp3 file.") from exc
    if x.shape[0] == 0:
        raise ScoringError("This file contains no audio.")
    # Average the channels (interleaved layout, as in the notebooks).
    x = U.to_mono(x.reshape(-1), x.shape[1]).astype(np.float32)
    return x, int(rate)


def to_16k(x, rate):
    """Resample mono audio to 16 kHz."""
    if rate == U.TARGET_RATE:
        return x
    if rate == U.SOURCE_RATE:
        return U.resample_48k_to_16k(x)
    from scipy.signal import resample_poly

    g = math.gcd(U.TARGET_RATE, rate)
    y = resample_poly(x, U.TARGET_RATE // g, rate // g)
    return y.astype(np.float32)


def _windows(speech):
    """Cut speech into windows; each is a dict with start_s, end_s, samples."""
    n, w = len(speech), U.CLIP_SAMPLES
    if n <= SINGLE_WINDOW_MAX:
        if n <= w:
            start, x = 0, speech
        else:
            start = (n - w) // 2  # centre crop, as in the notebooks
            x = U.fix_length(speech, w)
        spans = [(start, x)]
    else:
        starts = list(range(0, n - w + 1, HOP))
        if starts[-1] + w < n:  # make sure the end is covered
            starts.append(n - w)
        spans = [(s, speech[s:s + w]) for s in dict.fromkeys(starts)]
    rate = U.TARGET_RATE
    return [{"start_s": s / rate, "end_s": (s + len(x)) / rate,
             "samples": x} for s, x in spans]


def _prepare(data: bytes):
    """Prepare one recording; also returns its original duration (s)."""
    x, rate = decode(data)
    duration = len(x) / rate
    if duration > MAX_SECONDS:
        raise ScoringError(
            f"This recording is {duration:.0f} s long. The limit is "
            f"{MAX_SECONDS:.0f} s.")
    if float(np.abs(x).max()) < SILENCE_PEAK:
        raise ScoringError("This recording is silent.")
    x = to_16k(x, rate)
    speech, _, _ = U.trim_silence(x, U.TRIM_TOP_DB)
    speech = speech.astype(np.float32)
    return speech, _windows(speech), duration


def prepare(data: bytes):
    """Decode, resample, trim and window one recording.

    Returns
    -------
    tuple
        ``(speech_16k, windows)`` where each window is a dict with
        ``start_s``, ``end_s`` and ``samples``.
    """
    speech, windows, _ = _prepare(data)
    return speech, windows


# ------------------------------------------------------------------- #
# Encoder and head
# ------------------------------------------------------------------- #

class Encoder:
    """Frozen WavLM encoder; embeds one window per forward pass."""

    def __init__(self, feature_extractor, model):
        self.fe = feature_extractor
        self.model = model

    def embed(self, windows, layer):
        """Return mean-over-time hidden states of ``layer``, (n, 768)."""
        import torch

        out = []
        with torch.no_grad():
            for w in windows:
                inp = self.fe(w["samples"], sampling_rate=U.TARGET_RATE,
                              return_tensors="pt")
                h = self.model(inp["input_values"],
                               output_hidden_states=True).hidden_states
                out.append(h[layer][0].mean(dim=0).numpy())
        return np.stack(out).astype(np.float64)


def load_encoder():
    """Load the pinned WavLM model on CPU (downloads on first use).

    The UI wraps this in a cache so it runs once per session.
    """
    from transformers import AutoFeatureExtractor, AutoModel

    fe = AutoFeatureExtractor.from_pretrained(CHECKPOINT, revision=REVISION)
    model = AutoModel.from_pretrained(CHECKPOINT, revision=REVISION).eval()
    return Encoder(fe, model)


def embed(windows, encoder, layer):
    """Embed every window with ``encoder``; returns an (n, 768) array."""
    return np.asarray(encoder.embed(windows, layer), dtype=np.float64)


def head_proba(emb, head):
    """Class probabilities, shape (n, 8), columns in ``EMOTION_ORDER``."""
    z = ((np.atleast_2d(emb).astype(np.float64) - head["mean"])
         / head["scale"]) @ head["coef"].T + head["intercept"]
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    order = [head["classes"].index(e) for e in U.EMOTION_ORDER]
    return p[:, order]


# ------------------------------------------------------------------- #
# Scoring
# ------------------------------------------------------------------- #

def score_audio(data: bytes, assets: dict) -> dict:
    """Score one recording and apply the flag rule.

    Returns
    -------
    dict
        ``windows`` (each with start_s, end_s, probs, pred, confidence),
        ``probs`` (mean over windows, ``EMOTION_ORDER``), ``pred_emotion``,
        ``confidence``, ``flagged``, ``threshold_validated`` (True only
        for single-window calls, the case the threshold was tuned on),
        ``duration_s``, ``speech_s`` and ``warnings``.
    """
    speech, windows, duration = _prepare(data)
    warnings = []
    if len(speech) / U.TARGET_RATE < MIN_SPEECH_S:
        warnings.append("Less than 0.5 s of speech was found after "
                        "trimming silence; the result may be unreliable.")

    # Looked up on the module so a stub can replace it.
    encoder = sys.modules[__name__].load_encoder()
    head = assets["head"]
    emb = embed(windows, encoder, head["layer"])
    P = head_proba(emb, head)

    per_window = [{"start_s": w["start_s"], "end_s": w["end_s"],
                   "probs": p, "pred": U.EMOTION_ORDER[int(p.argmax())],
                   "confidence": float(p.max())}
                  for w, p in zip(windows, P)]
    probs = P.mean(axis=0)
    pred = U.EMOTION_ORDER[int(probs.argmax())]
    conf = float(probs.max())
    rule = assets["flag"]
    flagged = bool(pred in set(rule["negative_emotions"])
                   and conf >= float(rule["threshold"]))
    return {"windows": per_window, "probs": probs, "pred_emotion": pred,
            "confidence": conf, "flagged": flagged,
            "threshold_validated": len(windows) == 1,
            "duration_s": duration,
            "speech_s": len(speech) / U.TARGET_RATE,
            "warnings": warnings}

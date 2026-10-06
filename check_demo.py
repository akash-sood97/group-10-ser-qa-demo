"""Local checks for the QA demo engine; no model download needed.

Proves that the demo reproduces the notebooks: same preprocessing, same
head output, sensible windowing and limits, and an end-to-end run with
the cached embeddings standing in for the encoder.

Run:  python check_demo.py [--emb emb_wavlm.npy] [--data data/]
"""

import argparse
import io
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"  # an accidental download must fail

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scoring  # noqa: E402

U = scoring.U
FAILS = []


def find_up(rel):
    """Search upward from this script for a relative path."""
    for p in [HERE, *HERE.parents]:
        if (p / rel).exists():
            return p / rel
    return None


def report(name, ok, detail):
    """Print a PASS/FAIL line and remember failures."""
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        FAILS.append(name)


def wav_bytes(x, rate):
    """Encode a float array as 16-bit WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, x, rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def tone(seconds, rate, amp=0.3, hz=220.0):
    """A sine tone."""
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


class StubEncoder:
    """Returns preset embeddings instead of running WavLM."""

    def __init__(self):
        self.row = None  # (13, 768) array, or None to use ``fixed``
        self.fixed = None

    def embed(self, windows, layer):
        """One embedding per window."""
        v = self.fixed if self.row is None else self.row[layer]
        return np.tile(v, (len(windows), 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb", default=None, help="emb_wavlm.npy")
    ap.add_argument("--data", default=None, help="folder with data/")
    args = ap.parse_args()
    emb_path = Path(args.emb) if args.emb else (
        find_up("cache/ssl_embeddings/emb_wavlm.npy")
        or Path.home() / "Downloads/FP Final Submission/cache"
        / "ssl_embeddings/emb_wavlm.npy")
    data = Path(args.data) if args.data else (
        find_up("data/Person01") and find_up("data/Person01").parent
        or Path("/Users/akashsood/Desktop/ISB/Term 3/Foundation Project"
                "/Main Folder/data"))

    assets = scoring.load_assets(HERE / "assets")
    head = assets["head"]
    q = assets["queue"]
    index = U.load_index()
    E = np.load(emb_path)
    row_of = {p: i for i, p in enumerate(index["filepath"])}
    idx = np.array([row_of[p] for p in q["filepath"]])

    # (a) preprocessing parity with the notebook
    exact, one = 0, 0
    first_bad = None
    for p in q["filepath"]:
        path = data / p[len("data/"):]
        x = U.trimmed_signal(path)
        ref = U.fix_length(x, U.CLIP_SAMPLES) if len(x) > U.CLIP_SAMPLES else x
        _, wins = scoring.prepare(path.read_bytes())
        one += len(wins) == 1
        if len(wins) == 1 and np.array_equal(wins[0]["samples"], ref):
            exact += 1
        elif first_bad is None:
            first_bad = p
    report("(a) preprocessing parity", exact == 360 and one == 360,
           f"{exact}/360 exact, {one}/360 single window"
           + (f", first mismatch {first_bad}" if first_bad else ""))

    # (b) head parity with the saved held-out probabilities
    P = scoring.head_proba(E[idx, head["layer"], :], head)
    cols = [f"p_{e}" for e in U.EMOTION_ORDER]
    dp = float(np.abs(P - q[cols].to_numpy()).max())
    pred = np.array(U.EMOTION_ORDER)[P.argmax(axis=1)]
    agree = int((pred == q["pred_emotion"].to_numpy()).sum())
    report("(b) head parity", dp <= 1e-4 and agree == 360,
           f"{agree}/360 predictions, max|dp|={dp:.2e}")

    # (c) windowing and resampling
    sp, wins = scoring.prepare(wav_bytes(tone(20.0, 16000), 16000))
    starts = [round(w["start_s"] * 16000) for w in wins]
    ends = [round(w["end_s"] * 16000) for w in wins]
    regular = starts[:-1]
    ok = (len(wins) >= 10
          and regular == [28000 * i for i in range(len(regular))]
          and ends[-1] == len(sp)
          and all(e - s == 56000 for s, e in zip(starts, ends))
          and all(b - a <= 28000 for a, b in zip(starts, starts[1:]))
          and starts[0] == 0)
    report("(c1) 20 s windowing", ok,
           f"{len(wins)} windows, starts {[s / 16000 for s in starts[:4]]}"
           f"..., last ends at {ends[-1]}/{len(sp)}")
    sp4, w4 = scoring.prepare(wav_bytes(tone(4.0, 16000), 16000))
    report("(c2) 4 s tone", len(w4) == 1
           and len(w4[0]["samples"]) == 56000,
           f"{len(w4)} window of {len(w4[0]['samples'])} samples")
    sp44, _ = scoring.prepare(wav_bytes(tone(3.0, 44100), 44100))
    report("(c3) 44.1 kHz resample", abs(len(sp44) - 48000) <= 1 + 127,
           f"{len(sp44)} samples (3.0 s tone, trim may drop <128)")
    x44 = scoring.to_16k(tone(3.0, 44100), 44100)
    report("(c4) 44.1 kHz length", abs(len(x44) - 48000) <= 1,
           f"{len(x44)} samples vs 48000")

    # (d) limits
    for name, audio in [("70 s input", tone(70.0, 16000)),
                        ("all zeros", np.zeros(32000, np.float32))]:
        try:
            scoring.prepare(wav_bytes(audio, 16000))
            report(f"(d) {name}", False, "no error raised")
        except scoring.ScoringError as e:
            report(f"(d) {name}", True, f"ScoringError: {e}")

    # (e) end-to-end with a stub in place of WavLM
    stub = StubEncoder()
    real_loader = scoring.load_encoder
    scoring.load_encoder = lambda: stub
    try:
        man = assets["clips"]
        good = 0
        for r in man.itertuples():
            stub.row = E[row_of[r.filepath]]
            res = scoring.score_audio(
                (HERE / "assets" / "clips" / r.clip_file).read_bytes(),
                assets)
            good += (res["pred_emotion"] == r.saved_pred_emotion
                     and abs(res["confidence"] - r.saved_confidence) < 1e-3)
        report("(e1) bundled clips", good == len(man) == 16,
               f"{good}/{len(man)} reproduce the saved prediction")
        stub.row, stub.fixed = None, head["mean"]
        res = scoring.score_audio(wav_bytes(tone(20.0, 16000), 16000),
                                  assets)
        report("(e2) long call", len(res["windows"]) >= 10
               and res["threshold_validated"] is False,
               f"{len(res['windows'])} windows, threshold_validated="
               f"{res['threshold_validated']}")
    finally:
        scoring.load_encoder = real_loader

    # (f) the Streamlit app, every tab, with a stub in place of WavLM
    from streamlit.testing.v1 import AppTest

    scoring.load_encoder = lambda: stub
    try:
        at = AppTest.from_file(str(HERE / "app.py"),
                               default_timeout=60).run()
        report("(f1) app starts", not at.exception and len(at.tabs) == 3,
               f"{len(at.tabs)} tabs, exceptions: "
               f"{[e.value for e in at.exception]}")
        r = man.iloc[0]
        stub.row, stub.fixed = E[row_of[r.filepath]], None
        at.selectbox(key="sample_pick").select(r.clip_file)
        at.button(key="score_sample").click().run()
        texts = [m.value for m in at.markdown]
        line = [t for t in texts if "Saved prediction" in t]
        report("(f2) sample call scored",
               not at.exception and bool(line) and "✓" in line[0],
               line[0] if line else f"no saved-vs-live line; "
               f"{[e.value for e in at.exception]}")
    finally:
        scoring.load_encoder = real_loader

    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()

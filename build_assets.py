"""Build the data assets for the QA demo.

Refits the selected head (WavLM layer 5, scaler + logistic regression)
on the 1,080 development clips from cached embeddings, proves it
reproduces the saved held-out predictions, exports it as numpy arrays,
and writes the held-out review queue, sample clips and metric copies.
"""

import argparse
import datetime
import json
import re
import shutil
import sys
import wave
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SHARED = next(p / "shared" for p in [HERE, *HERE.parents]
              if (p / "shared" / "audio_utils.py").exists())
sys.path.insert(0, str(SHARED))

import audio_utils as U  # noqa: E402
import cv_harness as H  # noqa: E402
import speaker_split as S  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

LAYER = 5
ASSETS = HERE / "assets"
CLIPS = ASSETS / "clips"


def find_up(rel):
    """Search upward from this script for a relative path."""
    for p in [HERE, *HERE.parents]:
        if (p / rel).exists():
            return p / rel
    return None


def check(name, ok, detail):
    """Print a PASS/FAIL line; stop the build on failure."""
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        sys.exit(f"Check failed: {name}")


def softmax(z):
    """Row-wise softmax."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def n_channels(path):
    """Channel count from the WAV header."""
    with wave.open(str(path), "rb") as w:
        return w.getnchannels()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb", default=None, help="emb_wavlm.npy")
    ap.add_argument("--data", required=True, help="folder with Person01...")
    args = ap.parse_args()
    emb_path = Path(args.emb) if args.emb else find_up(
        "cache/ssl_embeddings/emb_wavlm.npy")
    data = Path(args.data)
    out = SHARED.parent / "Evaluation" / "outputs"
    tables = out / "tables"
    saved = pd.read_csv(out / "preds" / "wavlm_heldout.csv")
    CLIPS.mkdir(parents=True, exist_ok=True)

    # Load the index and embeddings, check shapes and the dev split.
    index = U.load_index()
    E = np.load(emb_path)
    check("embedding shape", E.shape == (1440, 13, 768), str(E.shape))
    dev = S.dev_mask(index)
    check("dev rows", int(dev.sum()) == 1080, str(int(dev.sum())))
    X = E[:, LAYER, :].astype(np.float64)
    y = index["emotion"].to_numpy()
    groups = index["speaker"].to_numpy()

    # Refit the head on development clips with the held-out settings.
    est = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced")
    pipe, _, _ = H.fit_tuned(est, X[dev], y[dev], groups[dev],
                             param_grid=None)
    classes = [str(c) for c in pipe.classes_]

    # Score held-out rows and compare with the saved predictions.
    ho = index[~dev].reset_index(drop=True)
    Xh = X[~dev]
    check("held-out filepaths",
          list(ho["filepath"]) == list(saved["filepath"]),
          f"{len(ho)} rows")
    proba = pipe.predict_proba(Xh)
    pred = np.array(classes)[proba.argmax(axis=1)]
    agree = int((pred == saved["pred_emotion"].to_numpy()).sum())
    pcols = [f"p_{e}" for e in U.EMOTION_ORDER]
    order = [classes.index(e) for e in U.EMOTION_ORDER]
    P = proba[:, order]
    dp = float(np.abs(P - saved[pcols].to_numpy()).max())
    check("head agreement", agree == 360 and dp <= 1e-4,
          f"{agree}/360, max|dp|={dp:.2e}")

    # Export the head as numpy arrays and verify a pure-numpy scorer.
    sc, clf = pipe.named_steps["scaler"], pipe.named_steps["clf"]
    np.savez(ASSETS / "wavlm_head.npz", mean=sc.mean_, scale=sc.scale_,
             coef=clf.coef_, intercept=clf.intercept_,
             classes=np.array(classes), layer=LAYER)
    z = ((Xh - sc.mean_) / sc.scale_) @ clf.coef_.T + clf.intercept_
    dn = float(np.abs(softmax(z) - proba).max())
    check("numpy head", dn <= 1e-6, f"max|dp|={dn:.2e}")

    # Write the head metadata.
    meta = {
        "checkpoint": "microsoft/wavlm-base-plus",
        "revision": "4c66d4806a428f2e922ccfa1a962776e232d487b",
        "licence": "CC BY-SA 3.0", "layer": LAYER, "C": 1.0,
        "n_train": int(dev.sum()),
        "pooling": "mean over time of hidden_states[5]",
        "input": "trimmed 30 dB, 16 kHz, <=3.5 s, not padded",
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (ASSETS / "head_meta.json").write_text(json.dumps(meta, indent=2))

    # Build the held-out queue with the flag rule.
    op = json.loads((tables / "flag_operating_point.json").read_text())
    neg, thr = set(op["negative_emotions"]), float(op["threshold"])
    q = pd.DataFrame({
        "filepath": ho["filepath"], "speaker": ho["speaker"],
        "true_emotion": ho["emotion"], "pred_emotion": pred,
        "confidence": proba.max(axis=1)})
    for i, c in enumerate(pcols):
        q[c] = P[:, i]
    q["truly_negative"] = q["true_emotion"].isin(neg)
    q["flagged"] = q["pred_emotion"].isin(neg) & (q["confidence"] >= thr)
    q = pd.concat([q[q["flagged"]].sort_values("confidence", ascending=False),
                   q[~q["flagged"]].sort_values("confidence",
                                                ascending=False)])
    q["queue_position"] = np.arange(1, len(q) + 1)
    q["clip_file"] = ""

    # Recompute the headline metrics and compare with the saved tables.
    fl, tn = q["flagged"], q["truly_negative"]
    rate = 100 * fl.mean()
    cap = 100 * (fl & tn).sum() / tn.sum()
    prec = 100 * (fl & tn).sum() / fl.sum()
    mf1 = f1_score(q["true_emotion"], q["pred_emotion"], average="macro")
    ev = pd.read_csv(tables / "summary_08.csv").set_index("H").loc[
        "H39", "evidence"]
    exp = [float(v) for v in re.findall(
        r"(?:rate|capture|precision) ([\d.]+)%", ev)]
    got = [round(rate, 1), round(cap, 1), round(prec, 1)]
    check("flag metrics", got == exp,
          f"rate {rate:.1f} / capture {cap:.1f} / precision {prec:.1f} "
          f"vs {exp}")
    hm = pd.read_csv(tables / "heldout_metrics.csv").set_index("model")
    exp_f1 = float(hm.loc["wavlm", "macro_f1"])
    check("macro-F1", round(mf1, 4) == round(exp_f1, 4),
          f"{mf1:.4f} vs {exp_f1}")

    # Pick two mono clips per emotion: first correct, first mispredicted.
    def mono(df):
        return [x for x in df.itertuples()
                if n_channels(data / x.filepath[len("data/"):]) == 1]

    man = []
    for e in U.EMOTION_ORDER:
        r = q[q["true_emotion"] == e].sort_values("filepath")
        ok = mono(r[r["pred_emotion"] == e])
        bad = mono(r[r["pred_emotion"] != e])
        pick = ok[:1] + (bad[:1] if bad else ok[1:2])
        check(f"clips {e}", len(pick) == 2, f"{len(pick)} mono clips")
        for x in pick:
            name = f"{x.speaker}_{Path(x.filepath).stem}.wav"
            shutil.copyfile(data / x.filepath[len("data/"):], CLIPS / name)
            q.loc[q["filepath"] == x.filepath, "clip_file"] = name
            man.append([name, x.filepath, x.speaker, x.true_emotion,
                        x.pred_emotion, round(x.confidence, 5)])
    pd.DataFrame(man, columns=["clip_file", "filepath", "speaker",
                               "true_emotion", "saved_pred_emotion",
                               "saved_confidence"]).to_csv(
        CLIPS / "manifest.csv", index=False)
    q.round(5).to_csv(ASSETS / "heldout_queue.csv", index=False)

    # Copy the metric files the demo displays.
    for f in ["heldout_metrics.csv", "heldout_recall_by_emotion.csv",
              "summary_08.csv", "flag_operating_point.json"]:
        shutil.copyfile(tables / f, ASSETS / f)
    print(f"DONE  {len(q)} queue rows, {len(man)} clips")


if __name__ == "__main__":
    main()

"""Streamlit interface for the call QA emotion review demo.

Three tabs: a review queue built from the held-out calls, a page to
analyse a new call (upload, record or bundled sample) and a page about
the model. Every reported number is read from ``assets/`` at run time.
"""

import hashlib
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import scoring

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
U = scoring.U  # shared chart conventions, found by scoring.py

# Page setup -----------------------------------------------------------
st.set_page_config(page_title="Group 10 · Call QA emotion review",
                   layout="wide")

# Encoder cache: wrap the current loader once so the model loads once per
# server process; a stub installed before the app runs is what is wrapped.
if not getattr(scoring.load_encoder, "_app_cached", False):
    _cached_loader = st.cache_resource(show_spinner=False)(
        scoring.load_encoder)
    try:
        _cached_loader._app_cached = True
    except AttributeError:  # wrapper refuses attributes; use a shim
        _inner = _cached_loader

        def _cached_loader():
            """Cached encoder loader."""
            return _inner()

        _cached_loader._app_cached = True
    scoring.load_encoder = _cached_loader

ORDER = list(U.EMOTION_ORDER)
COLOR_SCALE = alt.Scale(domain=ORDER,
                        range=[U.EMOTION_COLORS[e] for e in ORDER])
VERDICTS = ["—", "Agree", "Disagree"]


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_tables():
    """Read every table the interface shows from ``assets/``."""
    return {
        "queue": pd.read_csv(ASSETS / "heldout_queue.csv",
                             dtype={"speaker": str}),
        "metrics": pd.read_csv(ASSETS / "heldout_metrics.csv"
                               ).set_index("model"),
        "recall": pd.read_csv(ASSETS / "heldout_recall_by_emotion.csv"),
        "summary": pd.read_csv(ASSETS / "summary_08.csv"),
    }


@st.cache_data(show_spinner=False)
def load_assets_cached():
    """Head, metadata, flag rule and manifest, through the engine."""
    return scoring.load_assets(ASSETS)


@st.cache_data(show_spinner=False)
def clip_bytes(name):
    """Bytes of one bundled clip."""
    return (ASSETS / "clips" / name).read_bytes()


T = load_tables()
A = load_assets_cached()
RULE = A["flag"]
META = A["meta"]
NEG = list(RULE["negative_emotions"])
THR = float(RULE["threshold"])
Q = T["queue"]
M = T["metrics"]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def pct(x, digits=1):
    """Format a 0-1 fraction as a percentage string."""
    return f"{100 * x:.{digits}f}%"


def rule_words():
    """The flag rule in one sentence."""
    return (f"Flag a call when its top emotion is "
            f"{', '.join(NEG[:-1])} or {NEG[-1]} and the confidence is "
            f"at least {pct(THR)}.")


def badge(flagged):
    """HTML badge: accent colour when flagged."""
    if flagged:
        bg, fg, text = U.ACCENT, "#FFFFFF", "FLAGGED · hear this call first"
    else:
        bg, fg, text = U.MUTED, U.INK, "Not flagged · normal QA sampling"
    return (f"<span style='background:{bg};color:{fg};padding:4px 12px;"
            f"border-radius:12px;font-weight:600;font-size:0.9rem'>"
            f"{text}</span>")


def big_label(emotion, confidence):
    """Large predicted-emotion label in its colour."""
    return (f"<div style='font-size:2.2rem;font-weight:700;line-height:1.1;"
            f"color:{U.EMOTION_COLORS[emotion]}'>{emotion.capitalize()}"
            f"<span style='font-size:1.1rem;color:{U.GREY};"
            f"font-weight:500'> &nbsp;{pct(confidence)} confidence</span>"
            f"</div>")


def emotion_color(field="emotion", title="Emotion", legend=True):
    """Colour encoding shared by every chart."""
    return alt.Color(
        f"{field}:N", title=title, scale=COLOR_SCALE,
        legend=alt.Legend(orient="bottom", columns=8) if legend else None)


def show(chart, source=None):
    """Render a chart with its source caption."""
    st.altair_chart(chart.configure_view(strokeWidth=0), width="stretch")
    st.caption(source or U.SOURCE_NOTE)


def held_source():
    """Source caption for charts over the held-out queue."""
    return (f"{U.SOURCE_NOTE} · this chart: {len(Q)} held-out calls from "
            f"{Q['speaker'].nunique()} speakers")


CAT_AXIS = alt.Axis(labelOverlap=False, labelLimit=120)


def prob_chart(probs, title):
    """Eight-emotion probability bars."""
    df = pd.DataFrame({"emotion": ORDER,
                       "probability": np.asarray(probs) * 100})
    return alt.Chart(df, title=title).mark_bar().encode(
        y=alt.Y("emotion:N", sort=ORDER, title="Emotion", axis=CAT_AXIS),
        x=alt.X("probability:Q", title="Probability (%)",
                scale=alt.Scale(domain=[0, 100])),
        color=emotion_color(),
        tooltip=["emotion", alt.Tooltip("probability:Q", format=".1f")],
    ).properties(height=alt.Step(28))


def prob_title(probs):
    """Chart title stating the finding for one probability vector."""
    p = np.asarray(probs)
    i = int(p.argmax())
    runner = np.argsort(p)[-2]
    return (f"{ORDER[i].capitalize()} leads at {pct(p[i])}; "
            f"next is {ORDER[runner]} at {pct(p[runner])}")


def feedback_frame(base, store):
    """Add the reviewer columns to ``base`` from the session store."""
    out = base.copy()
    out["Reviewer verdict"] = [store.get(k, {}).get("verdict", "—")
                               for k in out.index]
    out["Reviewer emotion"] = [store.get(k, {}).get("emotion")
                               for k in out.index]
    out["Note"] = [store.get(k, {}).get("note", "") for k in out.index]
    return out


def feedback_editor(base, store_key, editor_key, filename_key):
    """Editable review table; returns nothing, updates the session store."""
    store = st.session_state.setdefault(store_key, {})
    frame = feedback_frame(base, store)
    locked = [c for c in frame.columns
              if c not in ("Reviewer verdict", "Reviewer emotion", "Note")]
    edited = st.data_editor(
        frame, key=editor_key, hide_index=True, disabled=locked,
        width="stretch",
        column_config={
            "Reviewer verdict": st.column_config.SelectboxColumn(
                "Reviewer verdict", options=VERDICTS, required=True),
            "Reviewer emotion": st.column_config.SelectboxColumn(
                "Reviewer emotion", options=ORDER),
            "Note": st.column_config.TextColumn("Note"),
            "Confidence (%)": st.column_config.NumberColumn(
                "Confidence (%)", format="%.1f"),
        })
    for key, row in edited.iterrows():
        emo = row["Reviewer emotion"]
        store[key] = {"verdict": row["Reviewer verdict"],
                      "emotion": None if pd.isna(emo) else emo,
                      "note": row["Note"] or ""}
    # Summary of the feedback given so far, over every row in the store.
    given = [v for v in store.values() if v["verdict"] != "—"]
    agree = sum(v["verdict"] == "Agree" for v in given)
    c1, c2 = st.columns(2)
    c1.metric("Calls reviewed", len(given))
    c2.metric("Reviewer agreement with the model",
              pct(agree / len(given)) if given else "—")
    log = pd.DataFrame(
        [{"call": k, **v} for k, v in store.items()
         if v["verdict"] != "—" or v["emotion"] or v["note"]],
        columns=["call", "verdict", "emotion", "note"])
    st.download_button("Download review log (CSV)",
                       log.to_csv(index=False).encode(),
                       file_name="Group-10_review_log.csv",
                       mime="text/csv", key=filename_key)


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------

st.title("Call QA emotion review")
st.markdown("**Decision supported:** which call a QA reviewer hears next.")
st.caption(
    f"Model: frozen {META['checkpoint']} speech encoder, layer "
    f"{META['layer']}, logistic-regression head · held-out macro-F1 "
    f"{M.loc['wavlm', 'macro_f1']:.4f} over {len(ORDER)} emotions.")

tab_queue, tab_call, tab_about = st.tabs(
    ["Review queue (held-out calls)", "Analyse a call", "About the model"])


# ----------------------------------------------------------------------
# Tab 1: review queue
# ----------------------------------------------------------------------

with tab_queue:
    n_all = len(Q)
    flagged = Q[Q["flagged"]]
    n_flag = len(flagged)
    n_neg = int(Q["truly_negative"].sum())
    n_hit = int(flagged["truly_negative"].sum())
    flag_rate = n_flag / n_all

    # KPI row.
    k = st.columns(5)
    k[0].metric("Calls in queue", n_all)
    k[1].metric("Flagged", n_flag, f"{pct(flag_rate)} flag rate",
                delta_color="off")
    k[2].metric("Negative calls captured", pct(n_hit / n_neg),
                help=f"{n_hit} of {n_neg} calls with a negative "
                     "audio-dataset label are in the flagged set.")
    k[3].metric("Precision of flags", pct(n_hit / n_flag),
                help=f"{n_hit} of {n_flag} flagged calls are negative.")
    k[4].metric("Random sample of the same size catches",
                pct(flag_rate),
                help="A random sample of this size would catch about "
                     "this share of the negative calls.")
    st.caption(rule_words() + " Flagged calls are heard first, ordered "
               "by confidence.")

    # Chart 1: the flagged queue in order.
    fq = flagged.assign(confidence_pct=flagged["confidence"] * 100,
                        emotion=flagged["pred_emotion"])
    lo_y = int(np.floor(THR * 100)) - 5
    dots = alt.Chart(
        fq, title=(f"{n_flag} flagged calls in queue order: confidence "
                   f"runs from {pct(fq['confidence'].max())} down to "
                   f"{pct(fq['confidence'].min())}")
    ).mark_circle(size=45, opacity=0.9).encode(
        x=alt.X("queue_position:Q", title="Queue position (1 = heard first)",
                scale=alt.Scale(domain=[0, int(fq["queue_position"].max()) + 1],
                                nice=False),
                axis=alt.Axis(labelOverlap=False)),
        y=alt.Y("confidence_pct:Q", title="Confidence (%)",
                scale=alt.Scale(domain=[lo_y, 100]),
                axis=alt.Axis(labelOverlap=False)),
        color=emotion_color(title="Predicted emotion"),
        tooltip=["queue_position", "emotion",
                 alt.Tooltip("confidence_pct:Q", format=".1f")],
    ).properties(height=260)
    rule = alt.Chart(pd.DataFrame({"y": [THR * 100]})).mark_rule(
        color=U.INK, strokeDash=[5, 3]).encode(y="y:Q")
    label = alt.Chart(pd.DataFrame(
        {"x": [int(fq["queue_position"].max()) + 1], "y": [THR * 100],
         "t": [f"threshold {pct(THR)}"]})
    ).mark_text(align="right", dx=-4, dy=-6, baseline="bottom",
                color=U.INK).encode(x="x:Q", y="y:Q", text="t:N")
    show(dots + rule + label, held_source())
    st.caption(f"The y-axis starts at {lo_y}% to show the spread of "
               f"confidence above the {pct(THR)} threshold.")

    # Chart 2 and 3 side by side: confusion heatmap and recall.
    left, right = st.columns(2)
    with left:
        ct = pd.crosstab(Q["true_emotion"], Q["pred_emotion"]
                         ).reindex(index=ORDER, columns=ORDER, fill_value=0)
        long = ct.stack().rename("n").reset_index()
        long.columns = ["true", "pred", "n"]
        # Label text and colour are decided here, not in a chart
        # expression, so the spec renders the same in every Vega-Lite
        # version. Both layers share this one table; empty cells get no
        # label.
        long["label"] = np.where(long["n"] > 0, long["n"].astype(str), "")
        long["ink"] = np.where(long["n"] > long["n"].max() / 2,
                               "white", U.INK)
        off = long[long["true"] != long["pred"]].sort_values(
            "n", ascending=False).iloc[0]
        base = alt.Chart(
            long, title=(f"Most common mix-up: {off['true']} heard as "
                         f"{off['pred']} ({int(off['n'])} calls)"))
        heat = base.mark_rect().encode(
            x=alt.X("pred:N", sort=ORDER, title="Predicted emotion",
                    axis=CAT_AXIS),
            y=alt.Y("true:N", sort=ORDER,
                    title="Audio dataset label (calls)", axis=CAT_AXIS),
            color=alt.Color("n:Q", title="Calls",
                            scale=alt.Scale(scheme="blues"),
                            legend=alt.Legend(orient="bottom")),
            tooltip=["true", "pred", "n"])
        text = base.mark_text().encode(
            x=alt.X("pred:N", sort=ORDER, axis=CAT_AXIS),
            y=alt.Y("true:N", sort=ORDER, axis=CAT_AXIS),
            text=alt.Text("label:N"),
            color=alt.Color("ink:N", legend=None,
                            scale=alt.Scale(domain=["white", U.INK],
                                            range=["white", U.INK])))
        st.altair_chart(
            (heat + text).properties(width=400, height=400
                                     ).configure_view(strokeWidth=0),
            width="content")
        st.caption(held_source())
    with right:
        rec = T["recall"][["true_emotion", "wavlm"]].rename(
            columns={"true_emotion": "emotion", "wavlm": "recall"})
        lo = rec.loc[rec["recall"].idxmin()]
        hi = rec.loc[rec["recall"].idxmax()]
        rchart = alt.Chart(
            rec, title=(f"{lo['emotion'].capitalize()} is recalled least "
                        f"({lo['recall']:.1f}%), {hi['emotion']} most "
                        f"({hi['recall']:.1f}%)")
        ).mark_bar().encode(
            y=alt.Y("emotion:N", sort=ORDER, title="Emotion", axis=CAT_AXIS),
            x=alt.X("recall:Q", title="Recall on held-out calls (%)",
                    scale=alt.Scale(domain=[0, 100])),
            color=emotion_color(),
            tooltip=["emotion", alt.Tooltip("recall:Q", format=".1f")],
        ).properties(height=alt.Step(28))
        show(rchart, held_source())

    # Filters.
    st.subheader("Queue")
    f1, f2, f3 = st.columns([1, 2, 1])
    scope = f1.radio("Show", ["Flagged only", "All calls"], horizontal=True,
                     key="q_scope")
    picked = f2.multiselect("Predicted emotion", ORDER, default=ORDER,
                            key="q_emotions")
    labels_on = f3.toggle("Show audio dataset labels", key="q_labels")
    view = Q if scope == "All calls" else flagged
    view = view[view["pred_emotion"].isin(picked)]

    # Queue table with reviewer feedback.
    base = pd.DataFrame({
        "Pos": view["queue_position"].to_numpy(),
        "▶": np.where(view["clip_file"].notna(), "▶", ""),
        "Predicted": view["pred_emotion"].to_numpy(),
        "Confidence (%)": (view["confidence"] * 100).round(1).to_numpy(),
        "Flag": np.where(view["flagged"], "Flagged", "—"),
    }, index=view["filepath"].to_numpy())
    if labels_on:
        base["Dataset label"] = view["true_emotion"].to_numpy()
        base["Result"] = np.where(
            view["true_emotion"] == view["pred_emotion"],
            "✓ correct", "✗ incorrect")
    st.caption("▶ marks the calls that carry audio. Edit the three right-"
               "hand columns to record your review.")
    feedback_editor(base, "fb_held",
                    f"ed_held_{scope}_{labels_on}_{'-'.join(picked)}",
                    "dl_held")

    # Call detail.
    st.subheader("Call detail")
    if view.empty:
        st.info("No calls match the filters.")
    else:
        opts = list(view["filepath"])
        names = {r.filepath: (f"#{r.queue_position} · {r.pred_emotion} "
                              f"{pct(r.confidence)}"
                              f"{' · ▶' if isinstance(r.clip_file, str) else ''}")
                 for r in view.itertuples()}
        sel = st.selectbox("Choose a call", opts, key="q_pick",
                           format_func=lambda p: names[p])
        row = Q[Q["filepath"] == sel].iloc[0]
        d1, d2 = st.columns([1, 2])
        with d1:
            st.markdown(big_label(row["pred_emotion"], row["confidence"]),
                        unsafe_allow_html=True)
            st.markdown(badge(bool(row["flagged"])), unsafe_allow_html=True)
            if labels_on:
                ok = row["true_emotion"] == row["pred_emotion"]
                st.markdown(f"Audio dataset label: **{row['true_emotion']}**"
                            f" · {'✓ correct' if ok else '✗ incorrect'}")
            if isinstance(row["clip_file"], str):
                st.audio(clip_bytes(row["clip_file"]), format="audio/wav")
            else:
                st.caption("No audio is bundled for this call.")
        with d2:
            probs = row[[f"p_{e}" for e in ORDER]].to_numpy(float)
            show(prob_chart(probs, prob_title(probs)), held_source())


# ----------------------------------------------------------------------
# Tab 2: analyse a call
# ----------------------------------------------------------------------

def score_one(call_id, name, source, data, saved=None):
    """Score one recording and add it to the session queue."""
    seen = st.session_state.setdefault("seen", set())
    if call_id in seen:
        return
    seen.add(call_id)
    try:
        with st.spinner("Loading the speech encoder (first time only, "
                        "~1 min)…"):
            res = scoring.score_audio(data, A)
    except scoring.ScoringError as exc:
        st.error(f"{name}: {exc}")
        return
    st.session_state.setdefault("calls", []).append(
        {"id": call_id, "name": name, "source": source, "res": res,
         "saved": saved})


def call_card(c):
    """Show the result for one scored call."""
    res = c["res"]
    with st.container(border=True):
        st.markdown(f"**{c['name']}** · {c['source']} · "
                    f"{res['duration_s']:.1f} s long, "
                    f"{res['speech_s']:.1f} s of speech")
        a, b = st.columns([1, 2])
        with a:
            st.markdown(big_label(res["pred_emotion"], res["confidence"]),
                        unsafe_allow_html=True)
            st.markdown(badge(res["flagged"]), unsafe_allow_html=True)
            st.caption(rule_words())
            if not res["threshold_validated"]:
                st.warning("Threshold calibrated on single 3.5 s clips; "
                           "applied here to window-averaged probabilities "
                           "— not validated.")
            for w in res["warnings"]:
                st.warning(w)
            if c["saved"] is not None:
                sp, sc = c["saved"]
                match = ("✓" if sp == res["pred_emotion"]
                         and abs(sc - res["confidence"]) < 1e-3 else "✗")
                st.markdown(
                    f"Saved prediction: {sp} ({sc:.2f}) · Live: "
                    f"{res['pred_emotion']} ({res['confidence']:.2f}) "
                    f"{match}")
        with b:
            live = (f"Live model scores for this call. {U.SOURCE_NOTE} "
                    "(data the model was built on).")
            show(prob_chart(res["probs"], prob_title(res["probs"])), live)
        wins = res["windows"]
        if len(wins) > 1:
            tl = pd.DataFrame({
                "start": [w["start_s"] for w in wins],
                "end": [w["end_s"] for w in wins],
                "emotion": [w["pred"] for w in wins],
                "confidence": [w["confidence"] * 100 for w in wins]})
            top = tl["emotion"].value_counts()
            tchart = alt.Chart(
                tl, title=(f"{top.index[0].capitalize()} is the top emotion"
                           f" in {int(top.iloc[0])} of {len(wins)} windows")
            ).mark_bar().encode(
                x=alt.X("start:Q", title="Time in call (seconds)",
                         axis=alt.Axis(labelOverlap=False)),
                x2="end:Q", y=alt.value(20),
                color=emotion_color(title="Predicted emotion"),
                opacity=alt.Opacity("confidence:Q", legend=alt.Legend(
                    title="Confidence (%)", orient="bottom"),
                    scale=alt.Scale(domain=[0, 100], range=[0.15, 1])),
                tooltip=["start", "end", "emotion",
                         alt.Tooltip("confidence:Q", format=".1f")],
            ).properties(height=70)
            show(tchart, live)
            area = pd.DataFrame(
                [{"time": (w["start_s"] + w["end_s"]) / 2, "emotion": e,
                  "probability": float(p) * 100}
                 for w in wins for e, p in zip(ORDER, w["probs"])])
            achart = alt.Chart(
                area, title=(f"Emotion mix over time: "
                             f"{res['pred_emotion']} leads on average at "
                             f"{pct(res['confidence'])}")
            ).mark_area().encode(
                x=alt.X("time:Q", title="Time in call (seconds)",
                        axis=alt.Axis(labelOverlap=False)),
                y=alt.Y("probability:Q", stack="zero",
                        title="Probability (%, stacked)",
                        axis=alt.Axis(labelOverlap=False)),
                color=emotion_color(),
                order=alt.Order("emotion:N", sort="ascending"),
                tooltip=["emotion", alt.Tooltip("probability:Q",
                                                format=".1f")],
            ).properties(height=28 * len(ORDER))
            show(achart, live)


with tab_call:
    st.markdown("Score a call you provide. Results join **My queue** "
                "below, ranked the same way as the held-out queue.")
    c1, c2, c3 = st.columns(3)
    with c1:
        uploads = st.file_uploader(
            "Upload recordings", type=["wav", "flac", "ogg", "mp3"],
            accept_multiple_files=True, key="uploads")
    with c2:
        rec_audio = st.audio_input("Record a voice", key="recording")
    with c3:
        clips = A["clips"]
        sample = st.selectbox(
            "Try a sample call", list(clips["clip_file"]), key="sample_pick",
            format_func=lambda f: (
                f"Person {f[:2]} · {Path(f[3:]).stem} · dataset label: "
                f"{clips.set_index('clip_file').loc[f, 'true_emotion']}"))
        go = st.button("Score sample call", key="score_sample")

    # Score new inputs (each is scored once per session).
    for up in uploads or []:
        data = up.getvalue()
        score_one(hashlib.md5(data).hexdigest() + up.name, up.name,
                  "upload", data)
    if rec_audio is not None:
        data = rec_audio.getvalue()
        score_one("rec:" + hashlib.md5(data).hexdigest(), "Recording",
                  "microphone", data)
    if go:
        row = clips.set_index("clip_file").loc[sample]
        score_one("sample:" + sample, f"Sample {sample}", "sample",
                  clip_bytes(sample),
                  (row["saved_pred_emotion"], float(row["saved_confidence"])))

    calls = st.session_state.get("calls", [])
    if not calls:
        st.info("No calls scored yet. Upload, record or pick a sample.")
    for c in reversed(calls):
        call_card(c)

    # Batch: every scored call, ranked like the held-out queue.
    if calls:
        st.subheader("My queue")
        ranked = sorted(calls, key=lambda c: (not c["res"]["flagged"],
                                              -c["res"]["confidence"]))
        mine = pd.DataFrame({
            "Rank": range(1, len(ranked) + 1),
            "Call": [c["name"] for c in ranked],
            "Source": [c["source"] for c in ranked],
            "Predicted": [c["res"]["pred_emotion"] for c in ranked],
            "Confidence (%)": [round(c["res"]["confidence"] * 100, 1)
                               for c in ranked],
            "Flag": ["Flagged" if c["res"]["flagged"] else "—"
                     for c in ranked],
            "Windows": [len(c["res"]["windows"]) for c in ranked],
        }, index=[c["id"] for c in ranked])
        feedback_editor(mine, "fb_mine", f"ed_mine_{len(calls)}", "dl_mine")


# ----------------------------------------------------------------------
# Tab 3: about the model
# ----------------------------------------------------------------------

with tab_about:
    st.subheader("Held-out results")
    st.caption(f"Scored on {Q['speaker'].nunique()} speakers the model "
               f"never saw ({n_all} calls), opened once.")
    m = st.columns(4)
    m[0].metric("Macro-F1, this model", f"{M.loc['wavlm', 'macro_f1']:.4f}",
                f"95% interval {M.loc['wavlm', 'ci_low']:.2f} to "
                f"{M.loc['wavlm', 'ci_high']:.2f}", delta_color="off")
    m[1].metric("Macro-F1, classical baseline",
                f"{M.loc['logreg', 'macro_f1']:.4f}")
    m[2].metric("Macro-F1, always the majority class",
                f"{M.loc['majority', 'macro_f1']:.4f}")
    m[3].metric("Accuracy, this model", pct(M.loc["wavlm", "accuracy"]))

    st.subheader("The flag rule")
    st.markdown(rule_words())
    dev_rate = float(RULE["flag_rate"])
    rule_tbl = pd.DataFrame({
        "": ["Development speakers", "Held-out speakers"],
        "Flag rate": [pct(dev_rate), pct(flag_rate)],
        "Capture of negative calls": [pct(RULE["dev_capture"]),
                                      pct(n_hit / n_neg)],
        "Precision of flags": [pct(RULE["dev_precision"]),
                               pct(n_hit / n_flag)],
    })
    st.dataframe(rule_tbl, hide_index=True, width="stretch")
    st.caption(f"Development figures: {RULE['source']}. Held-out figures "
               "are computed from the held-out queue.")
    sm = T["summary"][["hypothesis", "verdict", "evidence"]].rename(
        columns={"hypothesis": "Question", "verdict": "Verdict",
                 "evidence": "Evidence"})
    st.dataframe(sm, hide_index=True, width="stretch")

    st.subheader("How it fits the QA workflow")
    st.markdown(
        "- Flagged calls are heard first, ordered by confidence.\n"
        "- Unflagged calls go to normal QA sampling.\n"
        "- Reviewers record agree or disagree, which can be exported and "
        "used to recalibrate the threshold.")

    st.subheader("Caveats")
    n_train = int(META["n_train"])
    per_speaker = n_all // Q["speaker"].nunique()
    n_spk = n_train // per_speaker + Q["speaker"].nunique()
    st.markdown(
        f"- The audio dataset is acted speech from {n_spk} speakers.\n"
        f"- The model was trained on windows of up to {U.CLIP_SECONDS} s; "
        "longer calls are scored window by window and averaged.\n"
        "- Microphone or phone audio is out-of-domain and not validated.\n"
        "- The threshold should be recalibrated on pilot data.\n"
        f"- Held-out is {Q['speaker'].nunique()} speakers, so the "
        "interval is wide.")

    st.subheader("Credit")
    st.markdown(
        f"Encoder: `{META['checkpoint']}` (revision "
        f"`{META['revision'][:8]}`), licensed {META['licence']}. "
        "Used frozen; only the logistic-regression head was trained here.")

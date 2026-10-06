"""
Audio loading, preprocessing and chart conventions for the whole project.

Every notebook in ``Group-10/`` reads audio through this module,
so every model sees audio that was prepared in exactly the same way.
A second loader would drift from this one (a different trim threshold,
a different resampling filter) and results would stop being comparable.

What it provides
----------------
* Paths: where the audio dataset, the shared cache and each section's
  outputs live. Paths are found by searching upward, never hard-coded.
* The preprocessing chain used everywhere::

      read WAV -> mix to mono -> 48 kHz to 16 kHz -> trim silence
               -> centre to a fixed 3.5 s window

* The waveform cache (every clip, preprocessed once, stored as float16).
* Per-clip loudness and pitch measured on trimmed speech.
* Chart conventions: one colour per emotion, a fixed emotion order and
  a helper that puts title, axis labels, legend and source on a chart.

Dependencies: standard library, numpy and scipy. librosa and pandas are
imported only inside the functions that need them.
"""

from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import numpy as np
from scipy.signal import decimate

# ------------------------------------------------------------------- #
# Parameters shared by every section. Changing one changes every
# downstream number, so each carries the reason it has this value.
# ------------------------------------------------------------------- #

SOURCE_RATE = 48000  # every supplied file, checked in 01_EDA
TARGET_RATE = 16000  # 48000 / 3, so resampling is exact decimation
DECIMATE_Q = SOURCE_RATE // TARGET_RATE

CLIP_SECONDS = 3.5  # tested against 2.5 s and 3.0 s (H14)
CLIP_SAMPLES = int(CLIP_SECONDS * TARGET_RATE)  # 56,000 samples

TRIM_TOP_DB = 30.0  # tested against 40 dB (H13)
TRIM_FRAME = 512  # samples per frame when measuring energy
TRIM_HOP = 128

CACHE_DTYPE = np.float16  # halves the cache; no effect on features
SEED = 42

# The emotion code is the first field of every filename. The project
# brief gives this mapping and says to ignore the rest of the filename,
# so fields 2 to 4 are never decoded.
EMOTION_MAP = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}

N_CLIPS_EXPECTED = 1440
N_SPEAKERS_EXPECTED = 24
CLIPS_PER_SPEAKER_EXPECTED = 60

PITCH_FMIN, PITCH_FMAX = 60, 400  # Hz; the human voice range + margin


# ------------------------------------------------------------------- #
# Paths
# ------------------------------------------------------------------- #

def project_root() -> Path:
    """Return the folder that contains the audio dataset (``data/``).

    Walks upward from this file until it finds ``data/Person01``, so
    the submission works wherever it is unpacked, with no absolute
    paths anywhere.

    Returns
    -------
    Path
        The directory holding ``data/``.
    """
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / "data" / "Person01").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find the audio dataset. Place the folder 'data/' "
        "(containing Person01 ... Person24) beside 'Group-10/'."
    )


def data_root() -> Path:
    """Return the ``data/`` folder holding Person01 ... Person24."""
    return project_root() / "data"


def submission_root() -> Path:
    """Return ``Group-10/``, the parent of ``shared/``."""
    return Path(__file__).resolve().parent.parent


def _ensure(path: Path) -> Path:
    """Create a directory if it is missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def section_dir() -> Path:
    """Return the EDA and Feature engineering section folder.

    This section owns the tables every later section starts from:
    the label index, the acoustic measurements and the feature table.
    """
    return submission_root() / "EDA and Feature engineering"


def tables_dir() -> Path:
    """Return ``outputs/tables/`` of the EDA section."""
    return _ensure(section_dir() / "outputs" / "tables")


def features_dir() -> Path:
    """Return ``outputs/features/`` of the EDA section."""
    return _ensure(section_dir() / "outputs" / "features")


def figures_dir() -> Path:
    """Return ``outputs/figures/`` of the EDA section."""
    return _ensure(section_dir() / "outputs" / "figures")


def cache_dir() -> Path:
    """Return ``Group-10/cache/``.

    Holds files that are slow to build but can always be rebuilt from
    the audio (the waveform cache, intermediate feature blocks). It is
    not part of the upload.
    """
    return _ensure(submission_root() / "cache")


def cache_name(clip_seconds: float = None, top_db: float = None) -> str:
    """Return the waveform-cache file name for a window and trim."""
    clip_seconds = CLIP_SECONDS if clip_seconds is None else clip_seconds
    top_db = TRIM_TOP_DB if top_db is None else top_db
    return f"waves_{TARGET_RATE}_{clip_seconds}s_trim{int(top_db)}.npy"


CACHE_NAME = cache_name()


# ------------------------------------------------------------------- #
# The two variables: emotion (from the filename) and speaker (from
# the folder name)
# ------------------------------------------------------------------- #

def parse_filename(stem: str) -> dict:
    """Read the emotion label from a filename such as ``05-02-01-02-13``.

    Only field 1 is decoded. Fields 2 to 4 are checked for structure
    (five fields in total) but not interpreted, as the brief instructs.

    Parameters
    ----------
    stem : str
        File name without the ``.wav`` extension.

    Returns
    -------
    dict
        ``emotion_code`` (e.g. ``"05"``) and ``emotion`` (``"angry"``).
    """
    parts = stem.split("-")
    if len(parts) != 5:
        raise ValueError(f"Expected 5 filename fields, got {stem!r}")
    code = parts[0]
    return {"emotion_code": code,
            "emotion": EMOTION_MAP.get(code, "UNKNOWN")}


def speaker_from_folder(person_dir) -> str:
    """Return the speaker id from a folder name: ``Person07`` -> ``07``.

    The speaker groups the train/test split. It is never a model
    input, because a real call does not arrive labelled with who is
    speaking.
    """
    return str(Path(person_dir).name).replace("Person", "")


# ------------------------------------------------------------------- #
# Audio preprocessing, one step per function
# ------------------------------------------------------------------- #

def read_wav(path) -> tuple[np.ndarray, int, int]:
    """Read a 16-bit PCM WAV file with the standard library.

    Parameters
    ----------
    path : str or Path
        The ``.wav`` file.

    Returns
    -------
    tuple
        ``(samples, sample_rate_hz, n_channels)``. Samples are float32
        in [-1, 1], still interleaved if the file has two channels.
    """
    with wave.open(str(path), "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())
    if sampwidth != 2:
        raise ValueError(f"Expected 16-bit PCM, got {8 * sampwidth}-bit")
    x = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    return x / 32768.0, rate, n_channels


def to_mono(x: np.ndarray, n_channels: int) -> np.ndarray:
    """Average interleaved channels into one mono signal.

    Five supplied files are stereo. Read as mono they would decode as
    interleaved noise at twice their true length, so this step runs
    before anything else touches the audio.
    """
    if n_channels == 1:
        return x
    return x.reshape(-1, n_channels).mean(axis=1)


def resample_48k_to_16k(x: np.ndarray) -> np.ndarray:
    """Resample 48 kHz audio to 16 kHz.

    48,000 / 16,000 is exactly 3, so this is integer decimation: a
    low-pass filter against aliasing, then every third sample. The
    zero-phase FIR filter keeps the shape of the loudness envelope
    over time. 16 kHz keeps everything below 8 kHz, where the emotional
    content of speech lives, and makes every later step 3x cheaper.
    """
    if len(x) < 30:  # too short for the filter
        return x[::DECIMATE_Q]
    y = decimate(x, DECIMATE_Q, ftype="fir", zero_phase=True)
    return y.astype(np.float32)


def trim_silence(x: np.ndarray, top_db: float = TRIM_TOP_DB
                 ) -> tuple[np.ndarray, int, int]:
    """Remove leading and trailing near-silence.

    Splits the signal into frames, measures each frame's energy in dB
    relative to the loudest frame, and trims from both ends up to the
    first and last frame within ``top_db`` of that peak.

    Parameters
    ----------
    x : np.ndarray
        Mono signal.
    top_db : float
        Threshold in dB below the loudest frame.

    Returns
    -------
    tuple
        ``(trimmed_signal, start_sample, end_sample)``.
    """
    if len(x) < TRIM_FRAME:
        return x, 0, len(x)

    # Frame energy (RMS) for every hop across the clip.
    n_frames = 1 + (len(x) - TRIM_FRAME) // TRIM_HOP
    idx = (np.arange(TRIM_FRAME)[None, :]
           + TRIM_HOP * np.arange(n_frames)[:, None])
    rms = np.sqrt((x[idx] ** 2).mean(axis=1) + 1e-12)

    peak = rms.max()
    if peak <= 0:
        return x, 0, len(x)

    # Keep everything between the first and last frame that is
    # within top_db of the loudest frame.
    db = 20.0 * np.log10(rms / peak)
    loud = np.flatnonzero(db > -top_db)
    if loud.size == 0:
        return x, 0, len(x)
    start = int(loud[0] * TRIM_HOP)
    end = int(min(len(x), loud[-1] * TRIM_HOP + TRIM_FRAME))
    return x[start:end], start, end


def fix_length(x: np.ndarray, n: int = CLIP_SAMPLES) -> np.ndarray:
    """Force a clip to exactly ``n`` samples.

    Longer clips are centre-cropped and shorter ones centre-padded with
    zeros, so the speech sits in the middle of every window.
    """
    if len(x) == n:
        return x
    if len(x) > n:
        start = (len(x) - n) // 2
        return x[start:start + n]
    pad = n - len(x)
    return np.pad(x, (pad // 2, pad - pad // 2), mode="constant")


def load_clip(path, normalise: bool = False, clip_seconds: float = None,
              top_db: float = None) -> np.ndarray:
    """Return one clip, fully preprocessed and ready for a model.

    Runs the whole chain: read -> mono -> 16 kHz -> trim -> fixed
    window. Loudness is not normalised by default, because loudness is
    itself an emotional cue (see H10 and H11 in 01_EDA).

    Parameters
    ----------
    path : str or Path
        The ``.wav`` file.
    normalise : bool
        If True, scale the clip so its peak is 1.0.
    clip_seconds : float, optional
        Window length in seconds (default ``CLIP_SECONDS``).
    top_db : float, optional
        Trim threshold in dB (default ``TRIM_TOP_DB``).

    Returns
    -------
    np.ndarray
        float32 array of ``clip_seconds * 16000`` samples.
    """
    clip_seconds = CLIP_SECONDS if clip_seconds is None else clip_seconds
    top_db = TRIM_TOP_DB if top_db is None else top_db
    x, rate, ch = read_wav(path)
    x = to_mono(x, ch)
    if rate != TARGET_RATE:
        if rate != SOURCE_RATE:
            raise ValueError(f"Expected {SOURCE_RATE} Hz, got {rate} Hz")
        x = resample_48k_to_16k(x)
    x, _, _ = trim_silence(x, top_db=top_db)
    x = fix_length(x, int(round(clip_seconds * TARGET_RATE)))
    if normalise:
        peak = np.abs(x).max()
        if peak > 0:
            x = x / peak
    return x.astype(np.float32)


def trimmed_signal(path, top_db: float = None) -> np.ndarray:
    """Return the speech in a clip: mono, 16 kHz, trimmed, NOT padded.

    Use this to measure the audio (loudness, pitch, duration). Use
    ``load_clip()`` for model input, which needs a fixed length.
    """
    top_db = TRIM_TOP_DB if top_db is None else top_db
    x, rate, ch = read_wav(path)
    x = to_mono(x, ch)
    if rate == SOURCE_RATE:
        x = resample_48k_to_16k(x)
    x, _, _ = trim_silence(x, top_db=top_db)
    return x.astype(np.float32)


def waveform_md5(path) -> str:
    """Return the MD5 hash of a clip's decoded mono waveform.

    Two files with the same hash hold identical audio, whatever their
    names say.
    """
    x, _, ch = read_wav(path)
    return hashlib.md5(to_mono(x, ch).tobytes()).hexdigest()


# ------------------------------------------------------------------- #
# Tables built in 01_EDA and read by every later notebook
# ------------------------------------------------------------------- #

def load_index():
    """Return the label index: one row per clip, 1,440 rows.

    Columns include ``filepath``, ``speaker`` and ``emotion``. Row
    order is the order of every other per-clip table and array.
    """
    import pandas as pd

    path = tables_dir() / "label_index.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 01_EDA first.")
    return pd.read_csv(path, dtype={"speaker": str, "emotion_code": str})


def _cache_one(args):
    """Preprocess one clip for the cache (a worker process).

    Defined at module level because worker processes can only run
    functions they can import from a ``.py`` file.
    """
    i, rel, clip_seconds, top_db = args
    x = load_clip(project_root() / rel, clip_seconds=clip_seconds,
                  top_db=top_db)
    return i, x.astype(CACHE_DTYPE)


def build_waveform_cache(verbose: bool = True, clip_seconds: float = None,
                         top_db: float = None, n_workers: int = 8) -> Path:
    """Preprocess every clip once and save the result as a matrix.

    Writes ``cache/waves_16000_<window>s_trim<db>.npy`` (float16) and
    a manifest recording the exact parameters, so a stale cache can
    never be mistaken for a fresh one. Runs in parallel over
    ``n_workers`` processes (seconds on a recent 8-core machine).

    Returns
    -------
    Path
        The cache file.
    """
    import multiprocessing as mp

    clip_seconds = CLIP_SECONDS if clip_seconds is None else clip_seconds
    top_db = TRIM_TOP_DB if top_db is None else top_db
    n_samples = int(round(clip_seconds * TARGET_RATE))
    index = load_index()
    out = np.zeros((len(index), n_samples), dtype=CACHE_DTYPE)

    # Workers finish in any order; each result carries its row number.
    jobs = [(i, rel, clip_seconds, top_db)
            for i, rel in enumerate(index["filepath"])]
    with mp.Pool(n_workers) as pool:
        results = pool.imap_unordered(_cache_one, jobs, chunksize=16)
        for n, (i, x) in enumerate(results, 1):
            out[i] = x
            if verbose and n % 360 == 0:
                print(f"  {n}/{len(index)} clips")

    name = cache_name(clip_seconds, top_db)
    path = cache_dir() / name
    np.save(path, out)
    manifest = {
        "file": name,
        "n_clips": int(out.shape[0]),
        "n_samples": int(out.shape[1]),
        "target_rate": TARGET_RATE,
        "clip_seconds": clip_seconds,
        "trim_top_db": top_db,
        "dtype": CACHE_DTYPE.__name__,
        "normalised": False,
        "row_order": "matches outputs/tables/label_index.csv",
        "size_mb": round(path.stat().st_size / 1e6, 1),
    }
    with open(path.with_suffix(".manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    if verbose:
        print(f"wrote cache/{name}  {manifest['size_mb']} MB  "
              f"shape {out.shape}")
    return path


def load_waveform_cache(clip_seconds: float = None,
                        top_db: float = None) -> np.ndarray:
    """Return the waveform cache as float32, shape (1440, 56000).

    Row ``i`` is row ``i`` of ``load_index()``. Builds the cache first
    if it is missing.
    """
    path = cache_dir() / cache_name(clip_seconds, top_db)
    if not path.exists():
        build_waveform_cache(clip_seconds=clip_seconds, top_db=top_db)
    return np.load(path).astype(np.float32)


# ------------------------------------------------------------------- #
# Acoustic measurements, taken on trimmed speech and never on the
# padded cache. About half of each cached clip is zero padding, and
# the share differs per clip, so a loudness read off the cache would be
# diluted by a different amount for every clip.
# ------------------------------------------------------------------- #

ACOUSTICS_NAME = "acoustics.csv"


def _acoustics_one(args):
    """Measure one clip's speech length, loudness and pitch (a worker).

    Pitch uses pyin rather than yin because pyin marks which frames
    are voiced, so silence is kept out of the pitch estimate.
    """
    import librosa

    i, rel = args
    x = trimmed_signal(project_root() / rel)
    rms = float(np.sqrt((x ** 2).mean())) if len(x) else 0.0
    try:
        f0, voiced_flag, _ = librosa.pyin(
            x, fmin=PITCH_FMIN, fmax=PITCH_FMAX, sr=TARGET_RATE)
        voiced = f0[voiced_flag]
        f0_med = float(np.median(voiced)) if voiced.size else float("nan")
        voiced_frac = float(voiced_flag.mean())
    except Exception:
        f0_med, voiced_frac = float("nan"), float("nan")
    return i, len(x) / TARGET_RATE, rms, f0_med, voiced_frac


def build_acoustics_table(n_workers: int = 8, verbose: bool = True) -> Path:
    """Measure speech length, loudness and pitch for every clip.

    Writes ``outputs/tables/acoustics.csv`` with the columns filepath,
    speaker, emotion, trimmed_sec (s), trimmed_rms (linear 0-1),
    f0_median_voiced (Hz) and voiced_frac (0-1). pyin is slow, so this
    runs in parallel: 15 seconds to 5 minutes, depending on the machine.
    """
    import multiprocessing as mp

    import pandas as pd

    index = load_index()
    jobs = list(enumerate(index["filepath"]))
    with mp.Pool(n_workers) as pool:
        results = []
        stream = pool.imap_unordered(_acoustics_one, jobs, chunksize=16)
        for n, r in enumerate(stream, 1):
            results.append(r)
            if verbose and n % 360 == 0:
                print(f"  {n}/{len(jobs)} clips")
    results.sort()  # back into row order

    out = index[["filepath", "speaker", "emotion"]].copy()
    out["trimmed_sec"] = [r[1] for r in results]
    out["trimmed_rms"] = [r[2] for r in results]
    out["f0_median_voiced"] = [r[3] for r in results]
    out["voiced_frac"] = [r[4] for r in results]
    path = tables_dir() / ACOUSTICS_NAME
    out.to_csv(path, index=False)
    if verbose:
        n_f0 = out["f0_median_voiced"].notna().sum()
        print(f"wrote {path.name}  ({len(out)} rows, pitch on {n_f0})")
    return path


def load_acoustics():
    """Return the acoustics table, building it first if it is missing."""
    import pandas as pd

    path = tables_dir() / ACOUSTICS_NAME
    if not path.exists():
        build_acoustics_table()
    return pd.read_csv(path, dtype={"speaker": str})


# ------------------------------------------------------------------- #
# Chart conventions, shared so every chart in every document reads as
# one piece of work
# ------------------------------------------------------------------- #

# One colour per emotion, used by every chart.
EMOTION_COLORS = {
    "neutral": "#7C7389",
    "calm": "#4E9F8F",
    "happy": "#E8A33D",
    "sad": "#4B7BB5",
    "angry": "#C4453F",
    "fearful": "#8E5FA8",
    "disgust": "#6B8E3D",
    "surprised": "#D9738A",
}

# Fixed display order, low arousal to high arousal, so every chart
# reads the same way from left to right.
EMOTION_ORDER = ["neutral", "calm", "sad", "disgust",
                 "happy", "surprised", "fearful", "angry"]

# Highlighting: the marks that carry the inference use colour, and
# everything else is muted.
ACCENT = "#B0294A"  # call-outs and the key mark
GOOD = "#1F7A4C"  # "this is the property we want"
MUTED = "#C9C4D1"  # context marks
INK = "#1B1523"  # text and reference lines
GREY = "#7C7389"  # secondary text

SOURCE_NOTE = ("Source: own analysis of the supplied audio dataset, "
               "n=1,440 clips")


def finish_axes(ax, title: str, xlabel: str, ylabel: str,
                source: str = SOURCE_NOTE, legend: bool = False):
    """Apply the elements every chart needs.

    Title, labelled axes with units, a legend where there is more than
    one series, and a source line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The chart to finish.
    title, xlabel, ylabel : str
        The title should state the inference, not describe the chart.
    source : str
        Source line printed at the bottom left of the figure; pass an
        empty string on all but one panel of a multi-panel figure.
    legend : bool
        Draw the legend from the labelled series.
    """
    ax.set_title(title, fontsize=12, fontweight="600", loc="left", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    if legend:
        ax.legend(fontsize=9, frameon=False)
    if source:
        ax.figure.text(0.01, 0.005, source, fontsize=7.5, color=GREY,
                       ha="left")
    ax.figure.tight_layout(rect=[0, 0.02, 1, 1])


def save_figure(fig, name: str, out_dir: Path = None) -> Path:
    """Save a figure at slide resolution (200 dpi) and print its path.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        File name, e.g. ``"F01_class_distribution.png"``.
    out_dir : Path, optional
        Defaults to this section's ``outputs/figures/``.
    """
    out_dir = figures_dir() if out_dir is None else _ensure(Path(out_dir))
    path = out_dir / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved {path.relative_to(submission_root())}")
    return path

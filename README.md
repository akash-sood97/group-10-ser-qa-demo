# Group 10 · QA emotion review demo

A small web app built on the speech emotion model from the main package. It shows how the model's predictions could
help a quality-assurance reviewer decide which calls to hear first: it predicts one of 8 emotions for a recording,
gives a confidence, and flags the call for review when the confidence is low.

Live app: (link added after deployment)

## Run it locally

Python 3.12 or 3.13. From the folder that holds this README (or its parent, adjusting the paths):

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

If you are inside the `Group-10` package, the same commands work as
`pip install -r "QA Demo/requirements.txt"` and `streamlit run "QA Demo/app.py"`.

The first time you score a recording, the app downloads the public speech encoder (about 380 MB) once and keeps it
in the local cache. The review queue and the About tab work without it.

`requirements.txt` installs the CPU build of `torch`, so no GPU is needed. Its pins match
`Modelling/requirements.txt`, with one change: `pandas==2.3.3` instead of 3.0.2, because `streamlit==1.51.0`
requires pandas below 3. On a machine where the `--extra-index-url` line is not wanted, delete it; pip then installs
the default `torch` build.

## The tabs

| Tab | What it does |
|---|---|
| Review queue (held-out calls) | The calls from the six held-out speakers with predicted emotion, confidence and review flag, and columns to record a review |
| Analyse a call | Play one of the 16 bundled sample clips, upload a recording or record your voice, then score it: emotion, confidence, flag, and a comparison of the saved score with the live score |
| About the model | How the model was built, its held-out results, the flag rule and the caveats |

## The files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `scoring.py` | Scoring engine: decode, resample, trim, window, embed, classify, flag |
| `build_assets.py` | Builds everything in `assets/` from the main package's outputs |
| `check_demo.py` | Checks that the assets, the scoring engine and the app agree with the saved results |
| `assets/` | Classifier head, held-out queue and metrics tables, and the 16 sample clips |
| `.streamlit/config.toml` | Theme and the 20 MB upload limit |
| `shared/audio_utils.py` | The audio loader and preprocessing shared with the notebooks (found by searching upward from `app.py`) |

## How the assets were built and how to check them

Both scripts need the full `Group-10` package (its notebook outputs and `shared/` code), so run them from `Group-10/QA Demo/`, not from the stand-alone repo copy.

`build_assets.py` needs the frozen-encoder embedding cache written by the Modelling section of the main package
(`emb_wavlm.npy`) and the audio dataset folder:

```bash
python build_assets.py --emb path/to/emb_wavlm.npy --data path/to/data
```

To check the result:

```bash
python check_demo.py --emb path/to/emb_wavlm.npy --data path/to/data
```

The check needs the embedding cache and the audio dataset. It does not download the encoder.

## Caveats

- The audio dataset is acted speech from 24 speakers.
- The model was trained on windows of up to 3.5 s; longer calls are scored window by window and averaged.
- Microphone or phone audio is out-of-domain and not validated.
- The flag threshold should be recalibrated on pilot data.
- Held-out is 6 speakers, so the interval is wide.

## Credit

The encoder is `microsoft/wavlm-base-plus`, licensed CC BY-SA 3.0. It is used frozen; only the logistic-regression
head was trained here.

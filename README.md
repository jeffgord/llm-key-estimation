# LLM Key Detection

Code and data pipeline behind [*Using LLMs for Key Detection*](https://jeffgord.github.io/llm-key-detection/) — predicting the musical key of a song from a chord recognizer's output using an LLM, evaluated against DSP and deep learning baselines on FMAKv2.

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`ffmpeg`/`ffprobe` must also be on your `PATH` (used to trim audio).

The LLM methods (steps 5 below) call the Gemini API. Create a `.env` file in the repo root with:

```
GEMINI_API_KEYS=["your-api-key"]
```

You can list multiple keys (e.g. several free-tier AI Studio keys) — the scripts fall back to the next one whenever a key hits a rate limit, which is useful for getting through the full ~5,500-track dataset.

All commands below are run from the repo root and use their default input/output paths, so no flags are required for a basic run. Each script also accepts `--num-workers N` to parallelize, and a `run_*.sbatch` file is included alongside the slower steps for running on a Slurm cluster.

## Reproducing the results

1. **Download data** — FMAKv2 audio + key annotations, trimmed to 60s per the writeup:
   ```
   python data-prep/download_fma_keys.py
   python data-prep/trim_audio.py
   ```

2. **Chord recognition** — run the chord recognizer, then collapse its raw output into a compact per-track chord string:
   ```
   python chord-recognition/lvcr_predict.py
   python chords/preprocess.py
   ```

3. **Chroma features** — used by the KS baseline and by the Chroma-augmented LLM method:
   ```
   python chroma/extract.py
   python chroma/process.py
   ```

4. **Baselines**:
   ```
   python baseline/ks_predict.py
   python baseline/allconv_predict.py
   ```

5. **LLM methods** (requires `GEMINI_API_KEYS`):
   ```
   python method/llm_predict.py    # Chord-LLM
   python method2/llm_predict.py   # Chord-LLM w/ Chroma
   python method3/llm_predict.py   # Chord-LLM w/ Confidence (exploratory, not in final results)
   ```

6. **Evaluate** — run [`evaluation/eval.ipynb`](evaluation/eval.ipynb) top to bottom. It computes all metrics, builds the hybrid system, runs the significance tests, and generates the key-distance figures used in the writeup.

## Other folders

Not needed to reproduce the main results table:

- [`evaluation/dataset-stats.ipynb`](evaluation/dataset-stats.ipynb) — dataset size / genre diversity figures.
- `ismir-papers/download_ismir_abstracts.py` → `ismir-papers/classify_abstracts.py` → [`ismir-papers/analysis.ipynb`](ismir-papers/analysis.ipynb) — ISMIR paper-count-by-task figure.
- `poc/` — earlier proof-of-concept version of the same Chord-LLM pipeline, but run on symbolic chord data (Wikifonia/ChoCo lead sheets) instead of a chord recognizer's output.

## Updating the writeup site

The writeup is published to GitHub Pages via Quarto. After editing `writeup/writeup.qmd`, push the update from `main` with:

```
quarto publish gh-pages writeup/writeup.qmd
```

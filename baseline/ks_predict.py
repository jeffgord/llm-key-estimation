import csv
import sys
import argparse
import threading
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
import librosa
import numpy as np
import scipy

ROOTS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                          2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                          2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def predict_key(file_path: Path) -> str:
    y, sr = librosa.load(str(file_path))
    y_harm = librosa.effects.harmonic(y, margin=8)
    chroma_harm = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    chroma_filtered = librosa.decompose.nn_filter(
        chroma_harm, aggregate=np.median, metric="cosine"
    )
    chroma_nl_filtered = np.minimum(chroma_harm, chroma_filtered)
    chroma_smooth = scipy.ndimage.median_filter(chroma_nl_filtered, size=(1, 9))
    avg_chroma = np.mean(chroma_smooth, axis=1)

    best_correlation = -1.0
    best_key = None

    for i in range(12):
        rolled = np.roll(avg_chroma, -i)
        major_correlation = np.corrcoef(rolled, MAJOR_PROFILE)[0, 1]
        minor_correlation = np.corrcoef(rolled, MINOR_PROFILE)[0, 1]

        if major_correlation > best_correlation:
            best_correlation = major_correlation
            best_key = f"{ROOTS[i]} Major"
        if minor_correlation > best_correlation:
            best_correlation = minor_correlation
            best_key = f"{ROOTS[i]} Minor"

    return best_key


def load_completed(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    with open(output_path, newline="") as f:
        reader = csv.DictReader(f)
        return {int(row["track_id"]) for row in reader}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict musical key using the Krumhansl-Schmuckler algorithm")
    parser.add_argument("--data-dir", type=Path, default=Path("fma-keys"), help="Directory containing audio files")
    parser.add_argument("--output", type=Path, default=Path("baseline/ks-predictions.csv"), help="Output CSV path")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of worker threads (1 = no parallelism)")
    args = parser.parse_args()

    files = sorted(args.data_dir.rglob("*.mp3"))
    if not files:
        raise SystemExit(f"No .mp3 files found under {args.data_dir}")

    completed = load_completed(args.output)
    pending = [p for p in files if int(p.stem) not in completed]

    print(f"Total: {len(files)} files — {len(completed)} already done, {len(pending)} remaining")

    write_header = not args.output.exists() or args.output.stat().st_size == 0
    csv_lock = threading.Lock()

    def process_file(audio_path: Path) -> None:
        track_id = int(audio_path.stem)
        try:
            key = predict_key(audio_path)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}", file=sys.stderr)
            return
        with csv_lock:
            with open(args.output, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([track_id, key])

    with open(args.output, "a", newline="") as f:
        if write_header:
            csv.writer(f).writerow(["track_id", "key"])

    num_workers = max(1, args.num_workers or 1)

    if num_workers == 1:
        for p in tqdm(pending, desc="Predicting keys", unit="file"):
            process_file(p)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_file, p) for p in pending]
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(pending), desc="Predicting keys", unit="file"):
                pass

    print(f"Done. Results written to {args.output}")

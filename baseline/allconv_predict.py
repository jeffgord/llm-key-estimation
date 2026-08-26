import csv
import sys
import argparse
import threading
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
import madmom

_thread_local = threading.local()


def get_processor() -> madmom.features.key.CNNKeyRecognitionProcessor:
    if not hasattr(_thread_local, "proc"):
        _thread_local.proc = madmom.features.key.CNNKeyRecognitionProcessor()
    return _thread_local.proc


def predict_key(audio_path: Path) -> str:
    proc = get_processor()
    predictions = proc(str(audio_path))
    return madmom.features.key.key_prediction_to_label(predictions)


def load_completed(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    with open(output_path, newline="") as f:
        reader = csv.DictReader(f)
        return {int(row["track_id"]) for row in reader}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict musical key for audio files using madmom CNNKeyRecognitionProcessor")
    parser.add_argument("--data-dir", type=Path, default=Path("fma-keys"), help="Directory containing audio files")
    parser.add_argument("--output", type=Path, default=Path("baseline/allconv-predictions.csv"), help="Output CSV path")
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

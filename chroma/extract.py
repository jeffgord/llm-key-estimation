import sys
import argparse
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
import librosa
import numpy as np
import scipy


def extract_chroma(file_path: Path) -> np.ndarray:
    y, sr = librosa.load(str(file_path))
    y_harm = librosa.effects.harmonic(y, margin=8)
    chroma_harm = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    chroma_filtered = librosa.decompose.nn_filter(
        chroma_harm, aggregate=np.median, metric="cosine"
    )
    chroma_nl_filtered = np.minimum(chroma_harm, chroma_filtered)
    chroma_smooth = scipy.ndimage.median_filter(chroma_nl_filtered, size=(1, 9))
    return np.mean(chroma_smooth, axis=1)


def load_completed(output_dir: Path) -> set[int]:
    return {int(p.stem) for p in output_dir.glob("*.npy")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract average chroma vectors from audio files")
    parser.add_argument("--data-dir", type=Path, default=Path("fma-keys"), help="Directory containing audio files")
    parser.add_argument("--output-dir", type=Path, default=Path("chroma/raw"), help="Directory to write .npy files")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of worker threads (1 = no parallelism)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.data_dir.rglob("*.mp3"))
    if not files:
        raise SystemExit(f"No .mp3 files found under {args.data_dir}")

    completed = load_completed(args.output_dir)
    pending = [p for p in files if int(p.stem) not in completed]

    print(f"Total: {len(files)} files — {len(completed)} already done, {len(pending)} remaining")

    def process_file(audio_path: Path) -> None:
        track_id = int(audio_path.stem)
        try:
            chroma = extract_chroma(audio_path)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}", file=sys.stderr)
            return
        np.save(args.output_dir / f"{track_id}.npy", chroma)

    num_workers = max(1, args.num_workers or 1)

    if num_workers == 1:
        for p in tqdm(pending, desc="Extracting chroma", unit="file"):
            process_file(p)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_file, p) for p in pending]
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(pending), desc="Extracting chroma", unit="file"):
                pass

    print(f"Done. Vectors written to {args.output_dir}")

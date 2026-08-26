import csv
import argparse
from pathlib import Path
import numpy as np

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def format_chroma(vec: np.ndarray) -> str:
    normalized = vec / vec.sum()
    return ", ".join(f"{name}:{val:.3f}" for name, val in zip(PITCH_CLASSES, normalized))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw chroma vectors into a CSV")
    parser.add_argument("--raw-dir", type=Path, default=Path("chroma/raw"), help="Directory of .npy files")
    parser.add_argument("--output", type=Path, default=Path("chroma/chroma.csv"), help="Output CSV path")
    args = parser.parse_args()

    files = sorted(args.raw_dir.glob("*.npy"), key=lambda p: int(p.stem))
    if not files:
        raise SystemExit(f"No .npy files found under {args.raw_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "chroma_activations"])
        for path in files:
            track_id = int(path.stem)
            vec = np.load(path)
            writer.writerow([track_id, format_chroma(vec)])

    print(f"Wrote {len(files)} rows to {args.output}")

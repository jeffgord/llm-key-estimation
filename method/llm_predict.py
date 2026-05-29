import argparse
import csv
from pathlib import Path


def load_completed(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    with open(output_path, newline='') as f:
        reader = csv.DictReader(f)
        return {int(row['track_id']) for row in reader}


def make_estimator(model: str):
    if model == "gemini":
        from gemini_estimator import GeminiKeyEstimator
        return GeminiKeyEstimator()
    if model == "claude":
        from claude_estimator import ClaudeKeyEstimator
        return ClaudeKeyEstimator()
    raise ValueError(f"Unknown model {model!r}. Must be 'gemini' or 'claude'.")


def main():
    parser = argparse.ArgumentParser(description="Predict musical key using an LLM")
    parser.add_argument("--chords-csv", type=Path, default=Path("chords/chords.csv"))
    parser.add_argument("--output", type=Path, default=Path("method/llm-predictions.csv"))
    parser.add_argument("--provider", default="gemini", choices=["gemini", "claude"],
                        help="Which LLM provider to use")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Max predictions to make (-1 = unlimited)")
    args = parser.parse_args()

    with open(args.chords_csv, newline='') as f:
        rows = list(csv.DictReader(f))

    completed = load_completed(args.output)
    pending = [r for r in rows if int(r['track_id']) not in completed]

    if args.limit > 0:
        pending = pending[:args.limit]

    print(f"Total: {len(rows)} tracks — {len(completed)} already done, {len(pending)} to predict")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output.exists() or args.output.stat().st_size == 0
    with open(args.output, 'a', newline='') as f:
        if write_header:
            csv.writer(f).writerow(['track_id', 'key', 'explanation'])

    estimator = make_estimator(args.provider)

    from tqdm import tqdm
    with tqdm(pending, desc="Predicting keys", unit="track") as pbar:
        for row in pbar:
            track_id = int(row['track_id'])
            chords = row['chords']

            result = estimator.predict(chords)

            with open(args.output, 'a', newline='') as f:
                csv.writer(f).writerow([track_id, result.key, result.explanation])

    print(f"Done. Results written to {args.output}")


if __name__ == "__main__":
    main()

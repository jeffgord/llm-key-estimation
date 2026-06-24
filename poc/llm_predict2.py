import argparse
import csv
import json
from pathlib import Path

from dotenv import load_dotenv
import anthropic


class ClaudeKeyEstimator:
    def __init__(self):
        load_dotenv()
        self.client = anthropic.Anthropic(max_retries=5)
        self.model = "claude-sonnet-4-6"
        self.output_config = {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "explanation": {"type": "string"},
                    },
                    "required": ["key", "explanation"],
                    "additionalProperties": False,
                },
            }
        }

        prompt_path = Path(__file__).parent.parent / 'method' / 'prompt.txt'
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt = f.read()

    def predict(self, chords: str) -> str:
        contents = self.prompt.replace("{CHORDS}", chords)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=8000,
            thinking={"type": "adaptive"},
            output_config=self.output_config,
            messages=[{"role": "user", "content": contents}],
        )
        return next(b.text for b in response.content if b.type == "text")


def parse_response(text: str) -> dict:
    return json.loads(text)


def load_completed(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    with open(output_path, newline='') as f:
        reader = csv.DictReader(f)
        return {int(row['track_id']) for row in reader}


def main():
    parser = argparse.ArgumentParser(description="Predict musical key using Claude for the POC subset")
    parser.add_argument("--poc-csv", type=Path, default=Path(__file__).parent / "poc.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "llm-predictions2.csv")
    args = parser.parse_args()

    with open(args.poc_csv, newline='') as f:
        rows = list(csv.DictReader(f))

    completed = load_completed(args.output)
    pending = [r for r in rows if int(r['id']) not in completed]

    print(f"Total: {len(rows)} tracks — {len(completed)} already done, {len(pending)} remaining")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output.exists() or args.output.stat().st_size == 0
    with open(args.output, 'a', newline='') as f:
        if write_header:
            csv.writer(f).writerow(['track_id', 'key', 'explanation'])

    model = ClaudeKeyEstimator()

    from tqdm import tqdm
    with tqdm(pending, desc="Predicting keys", unit="track") as pbar:
        for row in pbar:
            track_id = int(row['id'])
            chords = row['chords']

            parsed = parse_response(model.predict(chords))
            key = parsed['key']
            explanation = parsed['explanation']

            with open(args.output, 'a', newline='') as f:
                csv.writer(f).writerow([track_id, key, explanation])

    print(f"Done. Results written to {args.output}")


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

class GeminiKeyEstimator:
    def __init__(self):
        load_dotenv()
        api_keys = json.loads(os.getenv('GEMINI_API_KEYS'))
        self.clients = [genai.Client(api_key=api_key) for api_key in api_keys]
        self.config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="high"),
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "key": types.Schema(type=types.Type.STRING),
                    "explanation": types.Schema(type=types.Type.STRING),
                },
                required=["explanation", "key"],
            ),
        )

        prompt_path = Path(__file__).parent / 'prompt.txt'
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt = f.read()

    def predict(self, chords: str) -> str:
        contents = self.prompt.replace("{CHORDS}", chords)
        num_clients = len(self.clients)

        for i in range(num_clients):
            try:
                response = self.clients[i].models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=contents,
                    config=self.config,
                )
                return response.text
            except errors.APIError as e:
                if e.code == 429 and i < num_clients - 1:
                    continue
                raise


def parse_response(text: str) -> dict:
    return json.loads(text)


def load_completed(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    with open(output_path, newline='') as f:
        reader = csv.DictReader(f)
        return {int(row['track_id']) for row in reader}


def main():
    parser = argparse.ArgumentParser(description="Predict musical key using Gemini LLM")
    parser.add_argument("--chords-csv", type=Path, default=Path("chords/chords.csv"))
    parser.add_argument("--output", type=Path, default=Path("method/llm-predictions.csv"))
    args = parser.parse_args()

    with open(args.chords_csv, newline='') as f:
        rows = list(csv.DictReader(f))

    completed = load_completed(args.output)
    pending = [r for r in rows if int(r['track_id']) not in completed]

    print(f"Total: {len(rows)} tracks — {len(completed)} already done, {len(pending)} remaining")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output.exists() or args.output.stat().st_size == 0
    with open(args.output, 'a', newline='') as f:
        if write_header:
            csv.writer(f).writerow(['track_id', 'key', 'explanation'])

    model = GeminiKeyEstimator()

    from tqdm import tqdm
    with tqdm(pending, desc="Predicting keys", unit="track") as pbar:
        for row in pbar:
            track_id = int(row['track_id'])
            chords = row['chords']

            parsed = parse_response(model.predict(chords))
            key = parsed['key']
            explanation = parsed['explanation']

            with open(args.output, 'a', newline='') as f:
                csv.writer(f).writerow([track_id, key, explanation])

    print(f"Done. Results written to {args.output}")


if __name__ == "__main__":
    main()

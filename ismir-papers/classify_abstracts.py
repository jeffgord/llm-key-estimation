# python ismir-papers/classify_abstracts.py

import argparse
import csv
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

TOPICS = [
    "lyrics-transcription",
    "audio-instrument-recognition",
    "audio-chord-estimation",
    "audio-beat-tracking",
    "audio-key-detection",
    "audio-melody-extraction",
    "multi-f0-estimation",
    "source-separation",
    "cover-song-identification",
    "NONE",
]


class GeminiAbstractClassifier:
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
                    "classification": types.Schema(type=types.Type.STRING, enum=TOPICS),
                },
                required=["classification"],
            ),
        )

        prompt_path = Path(__file__).parent / 'analyze-ismir-abstract.txt'
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt = f.read()

    def predict(self, title: str, year: str, abstract: str) -> str:
        contents = (
            self.prompt
            .replace("{TITLE}", title)
            .replace("{YEAR}", year)
            .replace("{ABSTRACT}", abstract)
        )
        num_clients = len(self.clients)

        for i in range(num_clients):
            while True:
                try:
                    response = self.clients[i].models.generate_content(
                        model="gemini-3.1-flash-lite",
                        contents=contents,
                        config=self.config,
                    )
                    return response.text
                except errors.APIError as e:
                    if e.code == 503:
                        time.sleep(30)
                        continue
                    if e.code == 429 and i < num_clients - 1:
                        break
                    raise


def parse_response(text: str) -> str:
    return json.loads(text)['classification']


def load_completed(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    with open(output_path, newline='') as f:
        reader = csv.DictReader(f)
        return {row['zenodo_id'] for row in reader}


def main():
    parser = argparse.ArgumentParser(description="Classify ISMIR abstracts by MIR topic using Gemini LLM")
    parser.add_argument("--abstracts-csv", type=Path, default=Path("ismir-papers/abstracts.csv"))
    parser.add_argument("--output", type=Path, default=Path("ismir-papers/classifications.csv"))
    args = parser.parse_args()

    with open(args.abstracts_csv, newline='') as f:
        rows = list(csv.DictReader(f))

    completed = load_completed(args.output)
    pending = [r for r in rows if r['zenodo_id'] not in completed]

    print(f"Total: {len(rows)} papers — {len(completed)} already done, {len(pending)} remaining")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output.exists() or args.output.stat().st_size == 0
    with open(args.output, 'a', newline='') as f:
        if write_header:
            csv.writer(f).writerow(['zenodo_id', 'classification'])

    model = GeminiAbstractClassifier()

    from tqdm import tqdm
    with tqdm(pending, desc="Classifying abstracts", unit="paper") as pbar:
        for row in pbar:
            classification = parse_response(
                model.predict(row['title'], row['year'], row['abstract'])
            )

            with open(args.output, 'a', newline='') as f:
                csv.writer(f).writerow([row['zenodo_id'], classification])

    print(f"Done. Results written to {args.output}")


if __name__ == "__main__":
    main()

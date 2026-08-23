# python data-prep/download_ismir_abstracts.py

import csv
import html
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen

YEARS = range(2020, 2026)
ZENODO_SEARCH_URL = "https://zenodo.org/api/records?q={query}&size=25&page={page}"
PAGE_DELAY = 1.0
MAX_RETRIES = 5


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_json_with_retry(url: str) -> dict | None:
    for attempt in range(MAX_RETRIES):
        try:
            with urlopen(url, timeout=30) as resp:
                return json.load(resp)
        except HTTPError as e:
            if e.code == 429 and attempt < MAX_RETRIES - 1:
                wait = int(e.headers.get("Retry-After", 5 * (attempt + 1)))
                time.sleep(wait)
                continue
            print(f"    HTTP {e.code} for {url}")
            return None
    return None


def fetch_year_records(year: int) -> list[dict]:
    query = quote(f'meeting.acronym:"ISMIR {year}"')
    hits = []
    page = 1
    while True:
        data = fetch_json_with_retry(ZENODO_SEARCH_URL.format(query=query, page=page))
        if not data:
            break
        page_hits = data["hits"]["hits"]
        if not page_hits:
            break
        hits.extend(page_hits)
        if len(hits) >= data["hits"]["total"]:
            break
        page += 1
        time.sleep(PAGE_DELAY)
    return hits


def dedupe_by_title(hits: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for h in hits:
        key = re.sub(r"[^a-z0-9]+", " ", h["metadata"]["title"].lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(h)
    return deduped


def download_ismir_abstracts(out_path: Path):
    rows = []
    for year in YEARS:
        print(f"ISMIR {year} ...")
        hits = dedupe_by_title(fetch_year_records(year))
        print(f"  {len(hits)} papers on Zenodo")

        for h in hits:
            description = h["metadata"].get("description")
            abstract = clean_text(description) if description else ""
            rows.append(
                {
                    "zenodo_id": h["id"],
                    "year": year,
                    "title": h["metadata"]["title"].strip(),
                    "abstract": abstract,
                }
            )

    missing = sum(1 for r in rows if not r["abstract"])
    print(f"Done. {len(rows)} papers total, {missing} missing an abstract.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["zenodo_id", "year", "title", "abstract"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    download_ismir_abstracts(Path("ismir-papers/abstracts.csv"))

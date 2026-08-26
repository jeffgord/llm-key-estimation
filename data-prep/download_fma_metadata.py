# python data-prep/download_fma_metadata.py

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

FMA_METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(downloaded / total_size * 100, 100)
    print(f"\r  {pct:.1f}% ({downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB)", end="", flush=True)


def download_fma_metadata(out_dir: Path):
    zip_path = Path("fma_metadata.zip")

    if zip_path.exists() and not zipfile.is_zipfile(zip_path):
        print(f"Existing {zip_path} is incomplete or corrupt, removing it.")
        zip_path.unlink()

    if not zip_path.exists():
        print("Downloading FMA metadata ...")
        urlretrieve(FMA_METADATA_URL, zip_path, reporthook=_progress)
        print()
    else:
        print(f"Archive already exists at {zip_path}, skipping download.")

    print(f"Extracting to {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    print("Cleaning up zip ...")
    zip_path.unlink()
    print("Done.")


if __name__ == "__main__":
    download_fma_metadata(Path("."))

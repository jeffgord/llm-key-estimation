# python data-prep/download_choco.py

import hashlib
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

CHOCO_URL = "https://zenodo.org/api/records/7706751/files/smashub%2Fchoco-v1.0.0.zip/content"
CHOCO_MD5 = "c26f23805e0f6beef7a0624df6be1ed0"
SUBSET = "smashub-choco-f7dd3ee/partitions/wikifonia/choco/jams-converted"


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(downloaded / total_size * 100, 100)
    print(f"\r  {pct:.1f}% ({downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB)", end="", flush=True)


def verify_md5(path: Path, expected: str) -> bool:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == expected


def download_choco(out_dir: Path):
    zip_path = Path("choco-v1.0.0.zip")

    if not zip_path.exists():
        print(f"Downloading ChoCo ...")
        urlretrieve(CHOCO_URL, zip_path, reporthook=_progress)
        print()
    else:
        print(f"Archive already exists at {zip_path}, skipping download.")

    print("Verifying MD5 ...", end=" ", flush=True)
    if not verify_md5(zip_path, CHOCO_MD5):
        raise ValueError(f"MD5 mismatch for {zip_path}. The file may be corrupted.")
    print("OK")

    print(f"Extracting wikifonia subset to {out_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.startswith(SUBSET)]
        for member in members:
            relative = Path(member).relative_to(SUBSET)
            target = out_dir / relative
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    print("Cleaning up zip ...")
    zip_path.unlink()
    print("Done.")


if __name__ == "__main__":
    download_choco(Path("wikifonia-choco"))

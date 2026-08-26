# python data-prep/download_data.py --subset

import argparse
from pathlib import Path
from urllib.request import urlretrieve
import mirdata

def download_fmak_audio(data_dir: Path, subset: bool = False):
    """Download the FMA dataset

    Args:
        subset (bool): Whether to download the small subset of the data (for testing)
    """
    fmak = mirdata.initialize("fma_keys", data_home=str(data_dir))

    if subset:
        fmak.download(partial_download=["tracks-000-019"])
    else:
        fmak.download()

def download_fmak_v2_annotations(data_dir: Path):
    """Download the FMA v2 annotations

    Args:
        data_dir (str): Directory to download the data to
    """
    annotations_url = "https://zenodo.org/api/records/12759100/files/fmakv2.csv/content"
    destination_path = data_dir / "fmakv2.csv"

    urlretrieve(annotations_url, destination_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the FMA dataset")
    parser.add_argument("--data_dir", type=Path, default=Path("fma-keys"), help="Directory to download the data to")
    parser.add_argument("--subset", action="store_true", help="Whether to download the small subset of the data (for testing)")
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    download_fmak_audio(args.data_dir, args.subset)
    download_fmak_v2_annotations(args.data_dir)
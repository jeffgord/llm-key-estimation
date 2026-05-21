# python data-prep/download_data.py --data_dir data --subset

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import mirdata

def download_fmak_audio(data_dir, subset=False):
    """Download the FMA dataset

    Args:
        data_dir (str): Directory to download the data to
        subset (bool): Whether to download the small subset of the data (for testing)
    """
    fmak = mirdata.initialize("fma_keys", data_home=data_dir)

    if subset:
        fmak.download(partial_download=["tracks-000-019"])
    else:
        fmak.download()

def download_fmak_v2_annotations(data_dir):
    """Download the FMA v2 annotations

    Args:
        data_dir (str): Directory to download the data to
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    annotations_url = "https://zenodo.org/api/records/12759100/files/fmakv2.csv/content"
    destination_path = data_dir / "fmakv2.csv"

    urlretrieve(annotations_url, destination_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the FMA dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to download the data to")
    parser.add_argument("--subset", action="store_true", help="Whether to download the small subset of the data (for testing)")
    args = parser.parse_args()

    download_fmak_audio(args.data_dir, args.subset)
    download_fmak_v2_annotations(args.data_dir)
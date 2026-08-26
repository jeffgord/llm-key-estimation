import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
import concurrent.futures


def duration_seconds(audio_path: Path) -> float:
    return float(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim MP3 files in place to a maximum length")
    parser.add_argument("--data-dir", type=Path, default=Path("fma-keys"), help="Directory containing audio files")
    parser.add_argument("--max-seconds", type=int, default=60, help="Maximum clip length")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of worker threads to use (1 = no parallelism)")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("ffmpeg and ffprobe must be installed and available on PATH")

    files = list(args.data_dir.rglob("*.mp3"))
    total_files = len(files)
    count = 0

    def process_file(source_path: Path) -> None:
        try:
            if duration_seconds(source_path) <= args.max_seconds:
                return
        except Exception:
            print(f"ffprobe error for {source_path}", file=sys.stderr)
            return

        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=source_path.suffix,
            dir=source_path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        str(source_path),
                        "-t",
                        str(args.max_seconds),
                        str(temp_path),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError as e:
                print(f"ffmpeg error for {source_path}:", e.stderr.decode(errors="replace"), file=sys.stderr)
                if temp_path.exists():
                    temp_path.unlink()
                return
            os.replace(temp_path, source_path)
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

    num_workers = args.num_workers if args.num_workers is not None else 1
    if num_workers < 1:
        num_workers = 1

    if num_workers == 1:
        for p in tqdm(files, desc=f"Trimming {args.data_dir}", unit="file"):
            process_file(p)
            count += 1
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_file, p) for p in files]
            for _ in tqdm(concurrent.futures.as_completed(futures), total=total_files, desc=f"Trimming {args.data_dir}", unit="file"):
                count += 1

    print(f"Processed {count} files in place under {args.data_dir}")
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import utils as u

MAX_SECONDS = 60

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
		).strip()
	)


if __name__ == "__main__":
	if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
		raise SystemExit("ffmpeg and ffprobe must be installed and available on PATH")

	count = 0

	for source_path in u.DATA_DIR.rglob("*.mp3"):
		if duration_seconds(source_path) <= MAX_SECONDS:
			count += 1
			continue

		with tempfile.NamedTemporaryFile(
			mode="wb",
			suffix=source_path.suffix,
			dir=source_path.parent,
			delete=False,
		) as temp_file:
			temp_path = Path(temp_file.name)

		try:
			subprocess.run(
				["ffmpeg", "-y", "-i", str(source_path), "-t", str(MAX_SECONDS), str(temp_path)],
				check=True,
			)
			os.replace(temp_path, source_path)
		finally:
			if temp_path.exists():
				temp_path.unlink()
		count += 1

	print(f"Processed {count} files in place under {u.DATA_DIR}")
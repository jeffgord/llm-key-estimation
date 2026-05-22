import sys
import argparse
import threading
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
import numpy as np

_ISMIR_DIR = Path(__file__).parent / 'ISMIR2019-Large-Vocabulary-Chord-Recognition'
sys.path.insert(0, str(_ISMIR_DIR))

from chordnet_ismir_naive import ChordNet
from mir.nn.train import NetworkInterface
from extractors.cqt import CQTV2
from mir import io, DataEntry
from extractors.xhmm_ismir import XHMMDecoder
from io_new.chordlab_io import ChordLabIO
from settings import DEFAULT_SR, DEFAULT_HOP_LENGTH

_DATA_DIR = _ISMIR_DIR / 'data'
_HMM_TEMPLATE = str(_DATA_DIR / 'submission_chord_list.txt')
MODEL_NAMES = ['joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s%d.best' % i for i in range(5)]

_thread_local = threading.local()


def get_nets_and_hmm():
    if not hasattr(_thread_local, 'nets'):
        _thread_local.nets = [
            NetworkInterface(ChordNet(None), name, load_checkpoint=False)
            for name in MODEL_NAMES
        ]
        _thread_local.hmm = XHMMDecoder(template_file=_HMM_TEMPLATE)
    return _thread_local.nets, _thread_local.hmm


def load_completed(output_dir: Path) -> set[int]:
    return {int(p.stem) for p in output_dir.glob('*.lab')}


def process_file(audio_path: Path, output_dir: Path) -> None:
    nets, hmm = get_nets_and_hmm()
    lab_path = output_dir / f'{int(audio_path.stem)}.lab'

    entry = DataEntry()
    entry.prop.set('sr', DEFAULT_SR)
    entry.prop.set('hop_length', DEFAULT_HOP_LENGTH)
    entry.append_file(str(audio_path.resolve()), io.MusicIO, 'music')
    entry.append_extractor(CQTV2, 'cqt')

    probs = [net.inference(entry.cqt) for net in nets]
    probs = [np.mean([p[i] for p in probs], axis=0) for i in range(len(probs[0]))]

    chordlab = hmm.decode_to_chordlab(entry, probs, False)
    entry.append_data(chordlab, ChordLabIO, 'chord')
    entry.save('chord', str(lab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run chord recognition on all audio files in a directory')
    parser.add_argument('--data-dir', type=Path, default=Path('fma-keys'), help='Directory containing audio files')
    parser.add_argument('--output-dir', type=Path, default=Path('chords'), help='Output directory for .lab files')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of worker threads')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.data_dir.rglob('*.mp3'))
    if not files:
        raise SystemExit(f'No .mp3 files found under {args.data_dir}')

    completed = load_completed(args.output_dir)
    pending = [p for p in files if int(p.stem) not in completed]
    print(f'Total: {len(files)} files — {len(completed)} already done, {len(pending)} remaining')

    def task(audio_path):
        try:
            process_file(audio_path, args.output_dir)
        except Exception as e:
            import sys
            print(f'Error processing {audio_path}: {e}', file=sys.stderr)

    if args.num_workers == 1:
        for p in tqdm(pending, desc='Recognizing chords', unit='file'):
            task(p)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(task, p) for p in pending]
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(pending), desc='Recognizing chords', unit='file'):
                pass

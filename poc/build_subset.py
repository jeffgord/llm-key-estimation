import jams
import jams.schema
import os
import pandas as pd

# ChoCo JAMS files use custom namespaces (e.g. timesig) unknown to the base library.
# namespace() checks `ns_key not in __NAMESPACE__` then indexes it, so we replace
# __NAMESPACE__ with a dict subclass that accepts any key.
class _PermissiveNamespaceDict(dict):
    def __contains__(self, _):
        return True
    def __missing__(self, key):
        return {"description": key, "dense": False}

jams.schema.__NAMESPACE__ = _PermissiveNamespaceDict(jams.schema.__NAMESPACE__)

wikifonia_choco_path = "wikifonia-choco/"

DEFAULT_BPM = 120


def _denominator_at_beat(beat, timesig_data):
    denominator = 4
    for obs in timesig_data:
        if float(obs.time) <= float(beat):
            denominator = obs.value.get("denominator", 4)
        else:
            break
    return denominator


def spb_at_beat(beat, timesig_data, bpm):
    return (60 / bpm) * (4 / _denominator_at_beat(beat, timesig_data))


def duration_in_seconds(start_beat, duration_beats, timesig_data, bpm):
    """Convert a beat-duration to seconds, segmenting across any time sig changes within the span."""
    start_beat = float(start_beat)
    end_beat = start_beat + float(duration_beats)

    breakpoints = [start_beat]
    for obs in timesig_data:
        t = float(obs.time)
        if start_beat < t < end_beat:
            breakpoints.append(t)
    breakpoints.append(end_beat)

    total = 0.0
    for i in range(len(breakpoints) - 1):
        seg_start = breakpoints[i]
        seg_end = breakpoints[i + 1]
        den = _denominator_at_beat(seg_start, timesig_data)
        total += (seg_end - seg_start) * (60 / bpm) * (4 / den)
    return total


def is_valid_chord_data(chord_data):
    return len(chord_data) > 0


def build_chord_repr(chords, timesig_data, bpm=DEFAULT_BPM, max_seconds=60.0):
    parts = []
    elapsed = 0.0
    for obs in chords:
        if elapsed >= max_seconds:
            break
        dur_sec = duration_in_seconds(obs.time, obs.duration, timesig_data, bpm)
        if elapsed + dur_sec > max_seconds:
            dur_sec = max_seconds - elapsed
        parts.append(f"{obs.value} {dur_sec:.1f}s")
        elapsed += dur_sec
    return ", ".join(parts)


def is_valid_key_data(key_data, timesig_data, bpm=DEFAULT_BPM):
    if len(key_data) == 0:
        return False
    if len(key_data) == 1:
        return True

    first = key_data[0]
    first_end_beat = float(first.time) + float(first.duration)
    first_dur_sec = duration_in_seconds(first.time, first.duration, timesig_data, bpm)

    if first_dur_sec < 60:
        return False
    return all(float(obs.time) >= first_end_beat for obs in key_data[1:])

def main():
    from tqdm import tqdm
    rows = []

    filenames = [f for f in os.listdir(wikifonia_choco_path) if f.endswith(".jams")]

    for filename in tqdm(filenames):
        track_id = filename.removesuffix(".jams").removeprefix("wikifonia_")
        file = os.path.join(wikifonia_choco_path, filename)
        jam = jams.load(file, strict=False)

        key_annotations = jam.search(namespace='key_mode')
        timesig_annotations = jam.search(namespace='timesig')
        chord_annotations = jam.search(namespace='chord_harte')

        if not key_annotations:
            continue

        key_data = key_annotations[0].data
        timesig_data = timesig_annotations[0].data if timesig_annotations else []

        if not is_valid_key_data(key_data, timesig_data):
            continue

        if not chord_annotations or not is_valid_chord_data(chord_annotations[0].data):
            continue

        chord_data = chord_annotations[0].data
        rows.append({
            "id": track_id,
            "key": key_data[0].value,
            "chords": build_chord_repr(chord_data, timesig_data),
        })

    df = pd.DataFrame(rows)
    if len(df) > 1000:
        df = df.sample(1000, random_state=4)
    df.to_csv("poc/poc.csv", index=False)
    print(f"Wrote {len(df)} tracks to poc.csv")

if __name__ == "__main__":
    main()

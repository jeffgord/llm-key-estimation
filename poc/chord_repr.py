import jams
import jams.schema
import jams.exceptions

# ChoCo JAMS files use custom namespaces unknown to the base library.
class _PermissiveNamespaceDict(dict):
    def __contains__(self, _):
        return True
    def __missing__(self, key):
        return {"description": key, "dense": False}

jams.schema.__NAMESPACE__ = _PermissiveNamespaceDict(jams.schema.__NAMESPACE__)

DEFAULT_BPM = 120


def spb_at_beat(beat, timesig_data, bpm):
    """Seconds per beat at a given beat position, accounting for time sig changes."""
    denominator = 4
    for obs in timesig_data:
        if float(obs.time) <= float(beat):
            denominator = obs.value.get("denominator", 4)
        else:
            break
    return (60 / bpm) * (4 / denominator)


def chord_repr(jam_path, bpm=DEFAULT_BPM):
    jam = jams.load(jam_path, strict=False)

    chord_anns = jam.search(namespace="chord_harte")
    timesig_anns = jam.search(namespace="timesig")

    if not chord_anns:
        return ""

    chords = chord_anns[0].data
    timesig = timesig_anns[0].data if timesig_anns else []

    parts = []
    for obs in chords:
        dur_sec = float(obs.duration) * spb_at_beat(obs.time, timesig, bpm)
        parts.append(f"{obs.value} {dur_sec:.1f}s")

    return "\n".join(parts)


if __name__ == "__main__":
    import os

    jams_dir = "wikifonia-choco"
    out_dir = "poc/chords"
    os.makedirs(out_dir, exist_ok=True)

    from tqdm import tqdm
    for filename in tqdm(os.listdir(jams_dir)):
        if not filename.endswith(".jams"):
            continue
        track_id = filename.removesuffix(".jams").removeprefix("wikifonia_")
        result = chord_repr(os.path.join(jams_dir, filename))
        if result:
            with open(os.path.join(out_dir, f"{track_id}.txt"), "w") as f:
                f.write(result + "\n")

    print(f"Done. Output in {out_dir}/")

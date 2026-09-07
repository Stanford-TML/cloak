#!/usr/bin/env python3
"""Reduce DROID metadata files from KarlP/droid into (LAB, timestamp)-keyed lookups,
and classify the per-episode wrist serial from per-serial focal length.

Three outputs:
  - assets/droid_lang_annotations_processed.json — language annotation triples
  - assets/droid_intrinsics_processed.json       — wrist+exterior camera intrinsics
  - assets/droid_zed_serials.json                — per-episode {"wrist": <serial>}

The raw annotations file is keyed by composite episode_id `<LAB>+<hash>+<timestamp>`.
The 8-char hash is opaque upstream state and exists in only two places: the
`metadata_<hash>.json` filename in the raw GCS bucket and the keys of this
annotations file. Some labs (notably all of BVL — 3,928 episodes) are absent
from the raw bucket entirely, so there is no way to recover the hash and match
on episode_id. We instead key on `(LAB, timestamp)`, which is unique for BVL
(zero collisions) and only collides for the 206 known TRI cases where upstream
uploaded two metadata files for the same trajectory.

The raw intrinsics file shares the same `<LAB>+<hash>+<timestamp>` composite
key scheme as the annotations file, so the same `(LAB, timestamp)` rekey
applies. Each per-camera value is a dict `{cameraMatrix, distCoeffs, width,
height}` where `cameraMatrix = [fx, cx, fy, cy]`, mostly at ZED HD (1280x720)
but a minority at other resolutions. We strip to just the camera matrix,
reorder to `[fx, fy, cx, cy]`, and rescale to RLDS resolution (320 wide)
using the source `width` field, with a centered-principal-point fallback
when `width` is missing or wrong (see `_scale_for_entry`).

The ZED-serials step is pure post-processing on the intrinsics map: across all
~50 unique physical cameras in DROID, ZED Mini (wrist) fx clusters near 183 px
at RLDS 320 width while ZED 2 (exterior) clusters near 133 px, with no entries
between. We classify each serial by mean fx and pick the wrist per episode.
Exteriors are not recorded — there's no way from intrinsics alone to tell which
physical exterior corresponds to which logical slot, and downstream consumers
only ever query the wrist.

Collision policy (annotations only):
  - Single annotation                 -> keep
  - Multiple, all identical triples   -> keep
  - Multiple, conflicting triples     -> drop (cannot disambiguate from the path)

Output JSONs are keyed by `<LAB>|<timestamp>` (pipe avoids collision with the
`+` separator used in the original episode_ids).

Usage:
    uv run python preprocessing/preprocess_metadata.py \\
        --annotations-path /tmp/droid_language_annotations.json \\
        --intrinsics-path /tmp/intrinsics.json
"""

from collections import Counter, defaultdict
import dataclasses
import json
from pathlib import Path
import statistics
import sys

import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils import make_lang_annotation_key, parse_annotation_episode_id  # noqa: E402

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
DEFAULT_ANN_OUTPUT = ASSETS_DIR / "droid_lang_annotations_processed.json"
DEFAULT_INTRINSICS_OUTPUT = ASSETS_DIR / "droid_intrinsics_processed.json"
DEFAULT_ZED_SERIALS_OUTPUT = ASSETS_DIR / "droid_zed_serials.json"


@dataclasses.dataclass
class Args:
    annotations_path: Path
    """Path to droid_language_annotations.json (downloaded from KarlP/droid)."""
    intrinsics_path: Path
    """Path to intrinsics.json (downloaded from KarlP/droid)."""
    annotations_output_path: Path = DEFAULT_ANN_OUTPUT
    """Output JSON path for processed language annotations."""
    intrinsics_output_path: Path = DEFAULT_INTRINSICS_OUTPUT
    """Output JSON path for processed camera intrinsics."""
    zed_serials_output_path: Path = DEFAULT_ZED_SERIALS_OUTPUT
    """Output JSON path for per-episode ZED serials."""


def _build_annotations(raw: dict) -> dict[str, list[str]]:
    """Build the (LAB|TS) -> triple lookup."""
    grouped: dict[str, list[list[str]]] = {}
    parse_failures: list[str] = []
    lab_counts: dict[str, int] = {}

    for episode_id, instrs in raw.items():
        parsed = parse_annotation_episode_id(episode_id)
        if parsed is None:
            parse_failures.append(episode_id)
            continue
        lab, year, month, day, hh, mm, ss = parsed
        # Annotations are stored as {language_instruction1, ..._2, ..._3}; flatten to a triple.
        if not isinstance(instrs, dict):
            parse_failures.append(episode_id)
            continue
        try:
            triple = [
                str(instrs["language_instruction1"]),
                str(instrs["language_instruction2"]),
                str(instrs["language_instruction3"]),
            ]
        except KeyError:
            parse_failures.append(episode_id)
            continue
        key = make_lang_annotation_key(lab, year, month, day, hh, mm, ss)
        grouped.setdefault(key, []).append(triple)
        lab_counts[lab] = lab_counts.get(lab, 0) + 1

    lookup: dict[str, list[str]] = {}
    n_unique = 0
    n_agree = 0
    n_conflict = 0
    conflict_examples: list[str] = []
    for key, triples in grouped.items():
        if len(triples) == 1:
            lookup[key] = triples[0]
            n_unique += 1
        elif all(t == triples[0] for t in triples):
            lookup[key] = triples[0]
            n_agree += 1
        else:
            n_conflict += 1
            if len(conflict_examples) < 5:
                conflict_examples.append(f"{key} :: {triples}")

    print()
    print("===== Annotations build summary =====")
    print(f"Total raw annotations           : {len(raw):,}")
    print(f"Parse failures                  : {len(parse_failures):,}")
    print(f"Unique (LAB,TS) keys            : {n_unique:,}")
    print(f"Collisions w/ identical triples : {n_agree:,}")
    print(f"Collisions dropped (conflicts)  : {n_conflict:,}")
    print(f"Final lookup entries            : {len(lookup):,}")
    print()
    print("Per-lab raw counts:")
    for lab in sorted(lab_counts):
        print(f"  {lab:<12} {lab_counts[lab]:>8,}")
    if parse_failures:
        print()
        print(f"WARN: first 5 parse failures (of {len(parse_failures)}):")
        for p in parse_failures[:5]:
            print(f"  {p}")
    if conflict_examples:
        print()
        print(f"WARN: first 5 dropped conflicts (of {n_conflict}):")
        for c in conflict_examples:
            print(f"  {c}")
    return lookup


RLDS_WIDTH = 320  # target resolution we project at downstream
# Plausible source widths upstream may have calibrated at. ZED native modes:
# VGA=672, HD720=1280, HD1080=1920, HD2K=2208 (per camera). Other entries (640, 960, 2560, 3840)
# are common downscales/upscales we've seen elsewhere. Anything outside this set is treated as
# "non-standard / wrong resolution" and we fall back to picking the scale that lands the
# principal point near the image center.
PLAUSIBLE_SOURCE_WIDTHS = {640, 672, 960, 1280, 1920, 2208, 2560, 3840}
CENTER_TOL_FRAC = 0.15  # |cx - W/2| / W <= this counts as a centered (sane) principal point


RLDS_HEIGHT = 180


def _centering_err(scale: float, cx: float, cy: float) -> float:
    """L1 distance of (cx, cy) at `scale` from the RLDS image center, normalized."""
    return abs(cx * scale - RLDS_WIDTH / 2) / RLDS_WIDTH + abs(cy * scale - RLDS_HEIGHT / 2) / RLDS_HEIGHT


def _scale_for_entry(fx: float, cx: float, fy: float, cy: float, width) -> tuple[float, str] | None:
    """Pick the per-entry scale to bring intrinsics to RLDS_WIDTH.

    Prefer the source `width` field, but only if it produces a centered principal
    point (sanity check — protects against the case where upstream stored the wrong
    `width` alongside intrinsics calibrated at a different resolution). Otherwise
    infer the scale by finding the value that lands (cx, cy) closest to the image
    center. Returns (scale, reason_tag), or None if no plausible width centers
    the principal point — caller should drop the entry.
    """
    if isinstance(width, (int, float)) and int(width) in PLAUSIBLE_SOURCE_WIDTHS:
        s = RLDS_WIDTH / float(width)
        if _centering_err(s, cx, cy) < CENTER_TOL_FRAC:
            return s, "from_width"
        # Source width given but the resulting principal point is way off-center —
        # treat the `width` field as unreliable and fall through to inference.

    # Try each plausible width, pick the one with the most centered principal point.
    # Catches entries whose `width` field is missing, wrong, or implausible.
    best_w, best_err = None, float("inf")
    for w in PLAUSIBLE_SOURCE_WIDTHS:
        err = _centering_err(RLDS_WIDTH / w, cx, cy)
        if err < best_err:
            best_err, best_w = err, w
    if best_err < CENTER_TOL_FRAC:
        return RLDS_WIDTH / best_w, "inferred"
    return None


def _build_intrinsics(raw: dict) -> dict[str, dict[str, list[float]]]:
    """Rekey the intrinsics file onto the (LAB|TS) namespace.

    Source per-camera value: {cameraMatrix: [fx, cx, fy, cy], distCoeffs, width, height}.
    The docstring claims 1280x720 but a small fraction of entries (notably some
    IRIS calibrations) are actually at other resolutions, so we derive the
    rescale-to-RLDS factor per-entry from the source `width` field, with a
    centered-principal-point fallback when `width` is missing/wrong.
    """
    out: dict[str, dict[str, list[float]]] = {}
    n_parse_fail = 0
    n_bad_shape = 0
    n_zero_cam = 0
    n_uncenterable = 0
    n_scale_from_width = 0
    n_scale_inferred = 0
    src_width_hist: Counter = Counter()  # source `width` field across `from_width` entries
    inferred_width_hist: Counter = Counter()  # picked width across `inferred` entries
    inferred_by_lab: Counter = Counter()  # which labs needed correction
    for episode_id, cams in raw.items():
        parsed = parse_annotation_episode_id(episode_id)
        if parsed is None:
            n_parse_fail += 1
            continue
        lab, year, month, day, hh, mm, ss = parsed
        key = make_lang_annotation_key(lab, year, month, day, hh, mm, ss)
        if not isinstance(cams, dict):
            n_bad_shape += 1
            continue
        per_episode: dict[str, list[float]] = {}
        for serial, payload in cams.items():
            cam_matrix = payload.get("cameraMatrix") if isinstance(payload, dict) else None
            if not (isinstance(cam_matrix, list) and len(cam_matrix) == 4):
                n_bad_shape += 1
                continue
            fx, cx, fy, cy = (float(x) for x in cam_matrix)
            # Upstream emits all-zero cameraMatrix for failed/missing calibrations
            # (~6% of entries). Treat as missing so the consumer falls back to lab defaults.
            if fx == 0.0 or fy == 0.0:
                n_zero_cam += 1
                continue
            src_width = payload.get("width")
            result = _scale_for_entry(fx, cx, fy, cy, src_width)
            if result is None:
                n_uncenterable += 1
                continue
            scale, tag = result
            if tag == "from_width":
                n_scale_from_width += 1
                src_width_hist[int(src_width)] += 1
            else:
                n_scale_inferred += 1
                inferred_width_hist[int(round(RLDS_WIDTH / scale))] += 1
                inferred_by_lab[lab] += 1
            per_episode[str(serial)] = [fx * scale, fy * scale, cx * scale, cy * scale]
        if per_episode:
            # If two source entries share a (LAB|TS) (TRI duplicate-metadata case),
            # the last write wins. The intrinsics across duplicates should match within
            # tolerance since the underlying physical setup is the same.
            out[key] = per_episode

    print()
    print("===== Intrinsics build summary =====")
    print(f"Total raw intrinsics entries    : {len(raw):,}")
    print(f"Parse failures                  : {n_parse_fail:,}")
    print(f"Malformed entries dropped       : {n_bad_shape:,}")
    print(f"All-zero cameras dropped        : {n_zero_cam:,}")
    print(f"Uncenterable cameras dropped    : {n_uncenterable:,}")
    print(f"Scale from source `width`       : {n_scale_from_width:,}")
    print(f"Scale inferred (centered pp)    : {n_scale_inferred:,}  <-- corrections")
    print(f"Final lookup entries            : {len(out):,}")
    if src_width_hist:
        print()
        print("Source `width` distribution (from_width entries):")
        for w, n in sorted(src_width_hist.items()):
            print(f"  {w:>5} px : {n:>7,}")
    if inferred_width_hist:
        print()
        print("Inferred source widths (corrections):")
        for w, n in sorted(inferred_width_hist.items()):
            print(f"  {w:>5} px : {n:>7,}")
    if inferred_by_lab:
        print()
        print("Corrections by lab (inferred):")
        for lab, n in sorted(inferred_by_lab.items(), key=lambda kv: -kv[1]):
            print(f"  {lab:<10} : {n:>7,}")
    return out


# ZED Mini (wrist) and ZED 2 (exterior) cameras have different focal lengths;
# at RLDS 320x180, ZED 2 fx clusters near ~133 px and ZED Mini fx clusters near
# ~183 px (verified across all 48 unique serials in the DROID intrinsics asset:
# 33 serials in [130, 140), 15 serials in [170, 190), with no entries between).
WRIST_FX_THRESHOLD = 150.0


def _build_zed_serials(intrinsics_lookup: dict[str, dict[str, list[float]]]) -> dict[str, dict]:
    """Per-episode wrist serial classified from per-serial mean fx in the intrinsics map.

    DROID has ~50 unique physical cameras across the dataset. The wrist (ZED Mini)
    has a longer focal length than the exteriors (ZED 2), so a single fx threshold
    cleanly partitions every serial into wrist vs exterior. Output is just
    `{(LAB|TS): {"wrist": <serial>}}` — exteriors are dropped because we have no
    way to disambiguate which physical exterior corresponds to which logical slot.
    """
    serial_fxs: dict[str, list[float]] = defaultdict(list)
    for cams in intrinsics_lookup.values():
        for serial, intr in cams.items():
            serial_fxs[serial].append(float(intr[0]))
    serial_is_wrist = {
        serial: statistics.fmean(fxs) > WRIST_FX_THRESHOLD for serial, fxs in serial_fxs.items()
    }

    out: dict[str, dict] = {}
    n_no_wrist = 0
    n_multi_wrist = 0
    multi_wrist_examples: list[str] = []
    no_wrist_examples: list[str] = []
    for key, cams in intrinsics_lookup.items():
        wrists = [s for s in cams if serial_is_wrist[s]]
        if not wrists:
            n_no_wrist += 1
            if len(no_wrist_examples) < 5:
                no_wrist_examples.append(key)
            continue
        if len(wrists) > 1:
            n_multi_wrist += 1
            if len(multi_wrist_examples) < 5:
                multi_wrist_examples.append(f"{key} -> {wrists}")
        # If multiple wrist-classified serials, take the one with highest fx
        # (most ZED-Mini-like). In practice this is a no-op since DROID rigs have 1 wrist.
        wrist = max(wrists, key=lambda s: statistics.fmean(serial_fxs[s]))
        out[key] = {"wrist": wrist}

    n_wrist_serials = sum(1 for is_w in serial_is_wrist.values() if is_w)
    n_exterior_serials = len(serial_is_wrist) - n_wrist_serials
    print()
    print("===== ZED-serial build summary =====")
    print(f"Intrinsics keys                 : {len(intrinsics_lookup):,}")
    print(f"Unique serials                  : {len(serial_is_wrist):,}")
    print(f"  classified as wrist (fx > {WRIST_FX_THRESHOLD:.0f}) : {n_wrist_serials}")
    print(f"  classified as exterior        : {n_exterior_serials}")
    print(f"Episodes with no wrist serial   : {n_no_wrist:,}")
    print(f"Episodes with >1 wrist serial   : {n_multi_wrist:,}")
    print(f"Final lookup entries            : {len(out):,}")
    if no_wrist_examples:
        print()
        print("First 5 episodes with no wrist serial:")
        for k in no_wrist_examples:
            print(f"  {k}")
    if multi_wrist_examples:
        print()
        print("First 5 episodes with multiple wrist-classified serials:")
        for s in multi_wrist_examples:
            print(f"  {s}")
    return out


def _write(out_path: Path, data: dict, label: str) -> None:
    out_path = out_path.expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))
    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {len(data):,} {label} entries -> {out_path} ({size_mb:.2f} MB)")


def main(args: Args) -> None:
    ann_path = args.annotations_path.expanduser()
    intr_path = args.intrinsics_path.expanduser()
    if not ann_path.exists():
        print(f"ERROR: annotations file not found: {ann_path}", file=sys.stderr)
        sys.exit(1)
    if not intr_path.exists():
        print(f"ERROR: intrinsics file not found: {intr_path}", file=sys.stderr)
        sys.exit(1)

    raw_ann = json.loads(ann_path.read_text())
    print(f"Loaded {len(raw_ann):,} raw annotations from {ann_path}")
    raw_intr = json.loads(intr_path.read_text())
    print(f"Loaded {len(raw_intr):,} raw intrinsics entries from {intr_path}")

    ann_lookup = _build_annotations(raw_ann)
    intr_lookup = _build_intrinsics(raw_intr)

    print()
    print("Sample annotation entries:")
    for k in list(ann_lookup)[:3]:
        print(f"  {k!r}: {ann_lookup[k]!r}")
    print()
    print("Sample intrinsics entries:")
    for k in list(intr_lookup)[:3]:
        print(f"  {k!r}: {intr_lookup[k]!r}")

    print()
    _write(args.annotations_output_path, ann_lookup, "annotation")
    _write(args.intrinsics_output_path, intr_lookup, "intrinsics")

    serials_lookup = _build_zed_serials(intr_lookup)
    _write(args.zed_serials_output_path, serials_lookup, "ZED-serial")


if __name__ == "__main__":
    main(tyro.cli(Args))

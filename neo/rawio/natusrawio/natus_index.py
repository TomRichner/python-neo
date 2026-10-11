"""
Recording layout of a Natus/XLTEK session: continuous segments and where their
samples are stored.

Port of ``xlteknatustomat/segment_builder.build_segments`` with one addition:
the sample rate and channel layout (recorded channels, discard bits, headbox type)
are read from every ``.erd`` header, and a change starts a new segment (a
discontinuity of zero length).

* Entries of all ``.etc`` files are sorted by samplestamp.
* Samples per entry: the difference of consecutive ``sample_num`` within one
  file (smaller than the stamp step when the file has a gap inside it); for the
  last entry of a file, the file's ``.stc`` sample count minus its
  ``sample_num``, else a full step bounded by the next file's first stamp.
* Two consecutive entries are continuous when the second stamp equals the
  first stamp plus its sample count, and both files have the same rate and
  channel layout.
* Segment start time from ``.snc`` (samplestamp interpolation at the segment's
  rate), else the ``.erd`` creation time.
* When a segment starts before the previous one ends, its start is trimmed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .natus_files import (
    ErdHeader,
    list_session_files,
    read_erd_header,
    read_etc_file,
    read_snc_file,
    read_stc_file,
)
from .erd_decode import ErdLayout

logger = logging.getLogger(__name__)


@dataclass
class ErdFile:
    path: Path
    size: int
    header: ErdHeader
    layout: ErdLayout
    sample_nums: np.ndarray  # int64, .etc local sample numbers (increasing)
    offsets: np.ndarray  # int64, .etc byte offsets into the .erd

    @property
    def stream_key(self):
        h = self.header
        # (rate, recorded channels, discard bits, physical channel of each recorded channel,
        # headbox type, physical channels): files with the same key share names and gains
        return (
            float(h.sample_freq),
            h.num_recorded_chans,
            h.discard_bits,
            tuple(h.rec_to_phys_map),
            h.headbox_type,
            h.num_channels,
        )


@dataclass
class Segment:
    rate: float
    stream_key: tuple
    start_stamp: int
    t_start_posix: float
    n_samples: int
    # (file index, first local sample, sample count), in order
    chunks: list[tuple[int, int, int]] = field(default_factory=list)


@dataclass
class _RawSegment:
    entries: list[int]  # indexes into the sorted entry list
    start_stamp: int
    total: int
    rate: float
    stream_key: tuple
    t_start: float = 0.0
    trimmed: int = 0


class NatusLayout:
    """Files, segments and sync table of one session folder."""

    def __init__(self, dirname):
        self.dirname = Path(dirname)
        listing = list_session_files(self.dirname)
        self.listing = listing
        if not listing.erd:
            raise FileNotFoundError(f"No .erd files in {self.dirname}")

        self.files: list[ErdFile] = []
        self._etc_tables: list[np.ndarray] = []
        for erd in listing.erd:
            etc = listing.etc.get(erd.stem.lower())
            if etc is None:
                logger.warning("No .etc file for %s, skipping it", erd.name)
                continue
            header = read_erd_header(erd)
            table = read_etc_file(etc)
            self.files.append(
                ErdFile(
                    path=erd,
                    size=listing.sizes[erd.name],
                    header=header,
                    layout=ErdLayout(header.num_channels, header.rec_to_phys_map),
                    sample_nums=table[:, 2].copy(),
                    offsets=table[:, 0].copy(),
                )
            )
            self._etc_tables.append(table)
        if not self.files:
            raise ValueError(f"No .erd file with an .etc index in {self.dirname}")

        self.sync = read_snc_file(listing.snc) if listing.snc is not None else None
        self.segments = self._build_segments()

    # ------------------------------------------------------------------
    def _stc_spans(self) -> dict[str, int]:
        if self.listing.stc is None:
            return {}
        try:
            return {e.segment_name.lower(): e.sample_span for e in read_stc_file(self.listing.stc)}
        except (OSError, ValueError) as e:
            logger.warning("Could not read %s (%s), using .etc only", self.listing.stc.name, e)
            return {}

    def _build_segments(self) -> list[Segment]:
        # 1. every .etc entry tagged with its file, sorted (stably) by samplestamp
        entries = []  # (file index, samplestamp, sample_num)
        for fi, table in enumerate(self._etc_tables):
            for row in table:
                entries.append((fi, int(row[1]), int(row[2])))
        entries.sort(key=lambda e: e[1])
        if not entries:
            raise ValueError(f"No .etc entries in {self.dirname}")
        n = len(entries)

        # 2. samples stored per entry
        spans = self._stc_spans()
        counts = [0] * n
        by_file: dict[int, list[int]] = {}
        for i, e in enumerate(entries):
            by_file.setdefault(e[0], []).append(i)
        for fi, idxs in by_file.items():
            steps = []
            for j in range(len(idxs) - 1):
                cur, nxt = entries[idxs[j]], entries[idxs[j + 1]]
                step = nxt[2] - cur[2]
                if step <= 0:
                    step = nxt[1] - cur[1]
                counts[idxs[j]] = step
                steps.append(step)
            typical = max(steps) if steps else 1000
            last = idxs[-1]
            span = spans.get(self.files[fi].path.stem.lower())
            remaining = span - entries[last][2] if span is not None else 0
            if 0 < remaining <= 10 * typical:
                counts[last] = remaining
            elif last + 1 < n:
                stamp_gap = entries[last + 1][1] - entries[last][1]
                counts[last] = min(typical, stamp_gap) if stamp_gap > 0 else typical
            else:
                counts[last] = typical

        # 3. continuity (and rate / channel layout changes)
        raw: list[_RawSegment] = []
        for i, (fi, stamp, _num) in enumerate(entries):
            key = self.files[fi].stream_key
            if i > 0:
                pfi, pstamp, _ = entries[i - 1]
                if stamp == pstamp + counts[i - 1] and self.files[pfi].stream_key == key:
                    raw[-1].entries.append(i)
                    raw[-1].total += counts[i]
                    continue
            raw.append(_RawSegment([i], stamp, counts[i], key[0], key))

        # 4. clock time of each segment start
        for seg in raw:
            t = self.sync.stamp_to_posix(seg.start_stamp, seg.rate) if self.sync is not None else None
            if t is None:
                hdr = self.files[entries[seg.entries[0]][0]].header
                if hdr.creation_time is not None:
                    t = hdr.creation_time.timestamp()
                    logger.warning("No .snc time for stamp %d, using the .erd creation time", seg.start_stamp)
                else:
                    t = 0.0
            seg.t_start = t

        # 5. overlaps: trim the start of the later segment
        for a, b in zip(raw[:-1], raw[1:]):
            a_end = a.t_start + a.total / a.rate
            if b.t_start < a_end:
                b.trimmed = int((a_end - b.t_start) * b.rate)
                logger.warning("Segments overlap by %.3f s, trimming %d samples", a_end - b.t_start, b.trimmed)

        # 6. final segments with their chunks
        segments = []
        for seg in raw:
            n_samples = seg.total - seg.trimmed
            if n_samples <= 0:
                continue
            chunks: list[list[int]] = []
            skip = seg.trimmed
            for i in seg.entries:
                fi, _stamp, num = entries[i]
                count = counts[i]
                if skip >= count:
                    skip -= count
                    continue
                num += skip
                count -= skip
                skip = 0
                if chunks and chunks[-1][0] == fi and chunks[-1][1] + chunks[-1][2] == num:
                    chunks[-1][2] += count
                else:
                    chunks.append([fi, num, count])
            segments.append(
                Segment(
                    rate=seg.rate,
                    stream_key=seg.stream_key,
                    start_stamp=seg.start_stamp + seg.trimmed,
                    t_start_posix=seg.t_start + seg.trimmed / seg.rate,
                    n_samples=n_samples,
                    chunks=[tuple(c) for c in chunks],
                )
            )
        if not segments:
            raise ValueError(f"No recorded data found in {self.dirname}")
        return segments

    # ------------------------------------------------------------------
    def stamp_rate(self, stamp: int) -> float:
        """Sample rate in force at a samplestamp (for .snc interpolation)."""
        rate = self.segments[0].rate
        for seg in self.segments:
            if seg.start_stamp <= stamp:
                rate = seg.rate
            else:
                break
        return rate

    def stamp_to_posix(self, stamp: int) -> float | None:
        if self.sync is None:
            return None
        return self.sync.stamp_to_posix(stamp, self.stamp_rate(stamp))

    def read_file_samples(self, file_index: int, start: int, stop: int) -> np.ndarray:
        """Decode local samples [start, stop) of one .erd, reading only the bytes needed."""
        from .erd_decode import decode_range

        f = self.files[file_index]
        count = stop - start
        if count <= 0:
            return np.zeros((0, f.layout.n_rec), dtype=np.int32)
        i0 = int(np.searchsorted(f.sample_nums, start, side="right")) - 1
        if i0 < 0:
            raise ValueError(f"Sample {start} is before the first index point of {f.path.name}")
        i1 = int(np.searchsorted(f.sample_nums, stop, side="left"))
        byte0 = int(f.offsets[i0])
        byte1 = int(f.offsets[i1]) if i1 < f.offsets.size else f.size
        buf = np.fromfile(f.path, dtype=np.uint8, count=byte1 - byte0, offset=byte0)
        out = decode_range(buf, f.layout, start - int(f.sample_nums[i0]), count)
        if out.shape[0] < count:
            logger.warning("%s holds fewer samples than indexed; padding with zeros", f.path.name)
            full = np.zeros((count, f.layout.n_rec), dtype=np.int32)
            full[: out.shape[0]] = out
            out = full
        return out

"""
RawIO for Natus/XLTEK (NeuroWorks) EEG session folders.

A session folder holds one or more ``.erd`` files (delta-compressed samples),
one ``.etc`` index per ``.erd``, and session-wide ``.stc`` (file table),
``.snc`` (clock sync) and ``.ent`` (notes, clips, montages) files.

Neo model:

* one block; one segment per continuous piece of recording. Gaps between files,
  gaps inside a file, and changes of sample rate or channel layout between
  files (a gap of zero length) all start a new segment.
* one signal stream per distinct (sample rate, channel layout), id such as
  ``"512Hz"``. A stream has samples only in the segments recorded with its
  rate; in the others its size is 0.
* signals are int32 counts. Amplifier (EEG) inputs have the gain
  ``8711 / (2**21 - 0.5) * 2**discard_bits`` and units uV; DC inputs (mV), oximetry
  (SpO2 in %, pulse rate in bpm), pleth (mV) and trigger channels have their own gain
  per headbox type (``natus_channels``). The kind of each channel is the signal array
  annotation ``natus_channel_kind``: "eeg" (any amplifier input, including ECG, EOG and
  EMG leads and unconnected inputs), "dc", "trigger", "oximetry", "pulse" or "pleth".
* channel names come from the last ``ChanNames`` list of the ``.ent``, mapped to the
  physical channels per headbox type (on a Quantum, the auxiliary channels DC1-DC16,
  TRIG, OSAT, PR and Pleth are named from the end of the list).
* ``.erd`` schemas other than 8 and 9 are refused (ValueError); a channel whose
  ``frequency_factor`` is not 32767 (recorded at another rate) is logged as a warning.
* time is in seconds from the start of the first segment (so segment 0 starts
  at 0.0); the absolute start is ``rec_datetime`` (UTC) in the block and
  segment annotations.
* event channels: ``clips`` (epochs: ``ClipNote``/``ClipNoteComment`` records
  with their FILETIME duration) and ``comments`` (events: ``Custom`` records),
  from the ``.ent`` file, each in the segment where it starts (or the last
  segment starting before it).

Reads decode only what they need: for samples [a, b) of one ``.erd`` the bytes
from the last ``.etc`` index point at or before a to the first index point at
or after b are read and decoded. Numba is required (``pip install neo[natus]``).

Author: Tom Richner
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..baserawio import (
    BaseRawIO,
    _signal_channel_dtype,
    _signal_stream_dtype,
    _signal_buffer_dtype,
    _spike_channel_dtype,
    _event_channel_dtype,
)
from .erd_decode import HAVE_NUMBA
from .natus_channels import channel_table
from .natus_files import (
    CLIP_TYPES,
    COMMENT_TYPE,
    parse_ent_records,
    parse_filetime_hex,
    parse_montage_channel_names,
)


class NatusRawIO(BaseRawIO):
    """
    Class for reading a Natus/XLTEK (NeuroWorks) EEG session folder.

    Parameters
    ----------
    dirname: str | Path
        The session folder (the one holding the ``.erd``, ``.etc``, ``.stc``,
        ``.snc`` and ``.ent`` files).

    Examples
    --------
    >>> import neo.rawio
    >>> r = neo.rawio.NatusRawIO(dirname="Patient~ Name_<guid>")
    >>> r.parse_header()
    >>> print(r)
    >>> raw = r.get_analogsignal_chunk(block_index=0, seg_index=0, i_start=0, i_stop=1024,
    ...                                stream_index=0, channel_indexes=range(10))
    >>> uv = r.rescale_signal_raw_to_float(raw, dtype="float32", stream_index=0,
    ...                                    channel_indexes=range(10))
    """

    extensions = ["erd"]
    rawmode = "one-dir"
    # Signal reads share no mutable state: the layout is built once in _parse_header, and
    # each read does its own np.fromfile and decodes with fresh state arrays. Callers may
    # run get_analogsignal_chunk from several threads at once.
    thread_safe_reads = True

    def __init__(self, dirname: str | Path = ""):
        if not HAVE_NUMBA:
            raise ImportError("NatusRawIO needs numba: pip install neo[natus]")
        self.dirname = str(dirname)
        BaseRawIO.__init__(self)

    def _source_name(self):
        return self.dirname

    # ------------------------------------------------------------------
    def _parse_header(self):
        from .natus_index import NatusLayout

        layout = NatusLayout(self.dirname)
        self._layout = layout
        segments = layout.segments
        self._segments = segments
        self._t0 = segments[0].t_start_posix

        ent_bytes = None
        if layout.listing.ent is not None:
            with open(layout.listing.ent, "rb") as f:
                ent_bytes = f.read()

        # streams, in order of first appearance
        stream_keys = []
        for seg in segments:
            if seg.stream_key not in stream_keys:
                stream_keys.append(seg.stream_key)
        self._stream_keys = stream_keys
        self._seg_stream = [stream_keys.index(seg.stream_key) for seg in segments]

        montage = parse_montage_channel_names(ent_bytes) if ent_bytes is not None else None
        self._montage_names = montage

        signal_streams = []
        signal_channels = []
        self._stream_kinds = []  # per stream: natus_channel_kind of each channel
        used_ids = set()
        for key in stream_keys:
            rate, n_rec, discard_bits, rec_to_phys, headbox_type, n_phys = key
            stream_id = f"{rate:g}Hz"
            k = 2
            while stream_id in used_ids:
                stream_id = f"{rate:g}Hz_{k}"
                k += 1
            used_ids.add(stream_id)
            signal_streams.append((f"{rate:g} Hz ({n_rec} ch)", stream_id, ""))
            table = channel_table(headbox_type, n_phys, discard_bits, montage)
            kinds = []
            for i, phys in enumerate(rec_to_phys):
                pname, kind, gain, units = table[phys - 1]
                if pname:
                    name = pname
                elif montage:
                    name = f"ch{phys:03d}"
                else:
                    name = f"ch{i + 1:03d}"
                kinds.append(kind)
                signal_channels.append((name, str(phys), rate, "int32", units, gain, 0.0, stream_id, ""))
            self._stream_kinds.append(kinds)

        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)
        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)
        self._stream_n_chan = [key[1] for key in stream_keys]

        event_channels = np.array(
            [("clips", "clips", "epoch"), ("comments", "comments", "event")], dtype=_event_channel_dtype
        )

        self.header = {
            "nb_block": 1,
            "nb_segment": [len(segments)],
            "signal_buffers": np.array([], dtype=_signal_buffer_dtype),
            "signal_streams": signal_streams,
            "signal_channels": signal_channels,
            "spike_channels": np.array([], dtype=_spike_channel_dtype),
            "event_channels": event_channels,
        }

        self._read_events(ent_bytes)

        self._generate_minimal_annotations()
        bl_ann = self.raw_annotations["blocks"][0]
        bl_ann["name"] = "Natus session"
        bl_ann["rec_datetime"] = datetime.fromtimestamp(self._t0, tz=timezone.utc)
        bl_ann["natus_rates"] = [key[0] for key in stream_keys]
        for seg_index, seg in enumerate(segments):
            seg_ann = bl_ann["segments"][seg_index]
            seg_ann["name"] = f"Segment {seg_index}"
            seg_ann["rec_datetime"] = datetime.fromtimestamp(seg.t_start_posix, tz=timezone.utc)
            seg_ann["natus_rate"] = seg.rate
            seg_ann["natus_stream_id"] = str(signal_streams[self._seg_stream[seg_index]]["id"])
            for stream_index, key in enumerate(stream_keys):
                sig_ann = seg_ann["signals"][stream_index]
                sig_ann["natus_headbox_type"] = key[4]
                sig_ann["__array_annotations__"]["natus_channel_kind"] = np.array(self._stream_kinds[stream_index])

    def _read_events(self, ent_bytes):
        """Clips and comments from the .ent, in seconds of the time base."""
        layout = self._layout
        annotations = []
        if ent_bytes is not None:
            for rec in parse_ent_records(ent_bytes, CLIP_TYPES | {COMMENT_TYPE}):
                data = rec.get("Data", {})
                data_type = data.get("Type", "") if isinstance(data, dict) else ""
                stamp = rec.get("Stamp")
                start = None
                if isinstance(stamp, (int, float)):
                    start = layout.stamp_to_posix(int(stamp))
                end = None
                if data_type in CLIP_TYPES and start is not None:
                    ft0 = parse_filetime_hex(data.get("StartTime", ""))
                    ft1 = parse_filetime_hex(data.get("EndTime", ""))
                    if ft0 is not None and ft1 is not None:
                        duration = (ft1 - ft0) / 1e7
                        if duration > 0:
                            end = start + duration
                annotations.append(
                    {
                        "kind": "clip" if data_type in CLIP_TYPES else "comment",
                        "name": str(rec.get("Text", "")).strip(),
                        "comment": str(rec.get("Comment", "")).strip(),
                        "data_type": data_type,
                        "start": start,
                        "end": end,
                        "origin": str(rec.get("Origin", "")),
                    }
                )
        annotations.sort(key=lambda a: (a["start"] is None, a["start"] or 0.0))
        # clip and comment rows with absolute (POSIX) times, for callers that want them
        self.natus_annotations = annotations

        seg_starts = np.array([s.t_start_posix - self._t0 for s in self._segments])
        n_seg = len(self._segments)
        # per event channel, per segment: times, durations, labels
        self._events = {
            ch: [{"times": [], "durations": [], "labels": []} for _ in range(n_seg)] for ch in range(2)
        }
        for a in annotations:
            if a["start"] is None:
                continue
            t = a["start"] - self._t0
            seg_index = max(int(np.searchsorted(seg_starts, t, side="right")) - 1, 0)
            ch = 0 if a["kind"] == "clip" else 1
            if ch == 0:
                duration = (a["end"] - a["start"]) if a["end"] is not None else 0.0
            else:
                duration = 0.0
            d = self._events[ch][seg_index]
            d["times"].append(t)
            d["durations"].append(duration)
            d["labels"].append(a["name"])
        for ch in range(2):
            for d in self._events[ch]:
                d["times"] = np.asarray(d["times"], dtype="float64")
                d["durations"] = np.asarray(d["durations"], dtype="float64")
                d["labels"] = np.asarray(d["labels"], dtype="U")

    # ------------------------------------------------------------------
    def _segment_t_start(self, block_index, seg_index):
        return self._segments[seg_index].t_start_posix - self._t0

    def _segment_t_stop(self, block_index, seg_index):
        seg = self._segments[seg_index]
        return seg.t_start_posix - self._t0 + seg.n_samples / seg.rate

    def _get_signal_size(self, block_index, seg_index, stream_index):
        if self._seg_stream[seg_index] != stream_index:
            return 0
        return self._segments[seg_index].n_samples

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return self._segment_t_start(block_index, seg_index)

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, stream_index, channel_indexes):
        size = self._get_signal_size(block_index, seg_index, stream_index)
        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = size
        if not 0 <= i_start <= i_stop <= size:
            raise IndexError(f"Samples [{i_start}, {i_stop}) out of range [0, {size}) for this segment and stream")

        n_chan = self._stream_n_chan[stream_index]
        if channel_indexes is None:
            channel_indexes = slice(None)
        n = i_stop - i_start
        if n == 0:
            return np.zeros((0, n_chan), dtype="int32")[:, channel_indexes]

        out = np.empty((n, n_chan), dtype="int32")
        pos = 0  # first segment sample of the current chunk
        filled = 0
        for file_index, local_start, count in self._segments[seg_index].chunks:
            a = max(i_start, pos)
            b = min(i_stop, pos + count)
            if a < b:
                out[filled : filled + b - a] = self._layout.read_file_samples(
                    file_index, local_start + a - pos, local_start + b - pos
                )
                filled += b - a
            pos += count
            if pos >= i_stop:
                break
        if isinstance(channel_indexes, slice) and channel_indexes == slice(None):
            return out
        return out[:, channel_indexes]

    # ------------------------------------------------------------------
    def _spike_count(self, block_index, seg_index, spike_channel_index):
        return 0

    def _get_spike_timestamps(self, block_index, seg_index, spike_channel_index, t_start, t_stop):
        return None

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        return None

    def _get_spike_raw_waveforms(self, block_index, seg_index, spike_channel_index, t_start, t_stop):
        return None

    def _event_count(self, block_index, seg_index, event_channel_index):
        return self._events[event_channel_index][seg_index]["times"].size

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        d = self._events[event_channel_index][seg_index]
        times, durations, labels = d["times"], d["durations"], d["labels"]
        keep = np.ones(times.size, dtype=bool)
        if t_start is not None:
            keep &= times >= t_start
        if t_stop is not None:
            keep &= times <= t_stop
        times = times[keep]
        labels = labels[keep]
        if self.header["event_channels"]["type"][event_channel_index] == b"epoch":
            durations = durations[keep]
        else:
            durations = None
        return times, durations, labels

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        return np.asarray(event_timestamps, dtype=dtype)

    def _rescale_epoch_duration(self, raw_duration, dtype, event_channel_index):
        return np.asarray(raw_duration, dtype=dtype)

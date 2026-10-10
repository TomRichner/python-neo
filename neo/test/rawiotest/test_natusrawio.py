"""
Tests of NatusRawIO on local Natus/XLTEK sample sessions.

The sample data is not public, so the tests are skipped unless these
environment variables are set:

* ``NATUS_TEST_ROOT``: folder holding ``Natus_deident1`` and ``Natus_deident2``
  (the four de-identified sample sessions).
* ``NATUS_DEV_ROOT`` (optional): folder holding the de-identified development
  sessions; used for the session whose sample rate changes between files.
* ``NATUS_REFERENCE_ROOT`` (optional): folder holding the reference
  implementations ``xlteknatustomat``, ``exportnatusevents`` and ``eegindex``
  (default ``~/Desktop/gitlab_repos``). Tests comparing against them are skipped
  when they cannot be imported.
"""

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from neo.test.rawiotest import rawio_compliance as compliance

SAMPLE_ROOT = os.environ.get("NATUS_TEST_ROOT")
DEV_ROOT = os.environ.get("NATUS_DEV_ROOT")
REFERENCE_ROOT = Path(os.environ.get("NATUS_REFERENCE_ROOT", Path.home() / "Desktop" / "gitlab_repos"))

SAMPLES = {
    "426dd159": "Natus_deident1/Xxxxx~ Xxxxxx_426dd159-4f07-4447-918c-564f55d82d13",
    "85ba4df1": "Natus_deident1/Xxxx~ Xxxx_85ba4df1-98ef-49fc-9e67-ae3d1f5ad0b3",
    "662aafc9": "Natus_deident2/Xxxxxx~ Xxxx_662aafc9-f1ff-4a91-bc55-0102a0e2db27",
    "42a53fa4": "Natus_deident2/Xxxxxxxx~ Xxxx_42a53fa4-f077-4d5b-bcb3-6446aecae1c1",
}
RATE_CHANGE_SESSION = "LastNameA~ FirstNameA_c0fe2d81-a1d2-4da9-818d-481ecbe0a4b1"

pytestmark = pytest.mark.skipif(not SAMPLE_ROOT, reason="NATUS_TEST_ROOT not set")

try:
    import numba  # noqa: F401

    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False


def _add_reference_paths():
    for sub in ("xlteknatustomat", "exportnatusevents/src", "eegindex/src"):
        p = REFERENCE_ROOT / sub
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _import_reference(name):
    _add_reference_paths()
    return pytest.importorskip(name)


_readers = {}


def get_reader(key):
    """One parsed reader per session for the whole module."""
    if not HAVE_NUMBA:
        pytest.skip("numba not installed")
    if key not in _readers:
        from neo.rawio import NatusRawIO

        if key == "c0fe2d81":
            if not DEV_ROOT:
                pytest.skip("NATUS_DEV_ROOT not set")
            d = Path(DEV_ROOT) / RATE_CHANGE_SESSION
        else:
            d = Path(SAMPLE_ROOT) / SAMPLES[key]
        if not d.is_dir():
            pytest.skip(f"session folder missing: {d}")
        r = NatusRawIO(dirname=d)
        r.parse_header()
        _readers[key] = r
    return _readers[key]


def seg_stream(reader, seg_index):
    sizes = [reader.get_signal_size(0, seg_index, s) for s in range(reader.signal_streams_count())]
    nonzero = [s for s, n in enumerate(sizes) if n > 0]
    assert len(nonzero) == 1
    return nonzero[0]


# ---------------------------------------------------------------------------
# reference sample values (xlteknatustomat)
# ---------------------------------------------------------------------------
# Channel-samples the (slow, pure Python) reference may decode from the file start;
# the first 20000 samples of a 276-channel session (5.5 M) stay under it.
REF_DIRECT_LIMIT = 6_000_000


def reference_file_samples(erd_path, start, stop, tmp_dir):
    """Samples [start, stop) of one .erd as (samples, channels), from erd_reader.

    Short reads decode from the file start. For long ones the reference decoder runs
    on a copy of the file holding its header and the bytes from the last .etc index
    point at or before ``start`` (every channel is absolute there).
    """
    erd_reader = _import_reference("erd_reader")
    etc_reader = _import_reference("etc_reader")
    header = erd_reader.read_erd_header(erd_path)
    if stop * header.num_recorded_chans <= REF_DIRECT_LIMIT:
        return erd_reader.read_erd_samples(erd_path, header, max_samples=stop)[:, start:stop].T
    entries = etc_reader.read_etc_file(Path(erd_path).with_suffix(".etc"))
    entry = [e for e in entries if e.sample_num <= start][-1]
    after = [e.offset for e in entries if e.sample_num >= stop]
    with open(erd_path, "rb") as f:
        head = f.read(erd_reader.SAMPLE_PACKET_OFFSET)
        f.seek(entry.offset)
        body = f.read(after[0] - entry.offset) if after else f.read()
    tmp = Path(tmp_dir) / "piece.erd"
    tmp.write_bytes(head + body)
    out = erd_reader.read_erd_samples(tmp, max_samples=stop - entry.sample_num)
    return out[:, start - entry.sample_num :].T


def reference_segments(session_dir):
    """build_segments() output with the overlap trim applied, as eegindex uses it."""
    erd_reader = _import_reference("erd_reader")
    segment_builder = _import_reference("segment_builder")
    erds = erd_reader.get_erd_files_from_data_dir(session_dir)
    rate = erd_reader.read_erd_header(erds[0]).sample_freq
    out = []
    for seg in segment_builder.build_segments(session_dir, rate):
        n = seg.total_samples - seg.trimmed_samples
        if n <= 0:
            continue
        chunks, skip = [], seg.trimmed_samples
        for c in seg.chunks:
            if skip >= c.sample_count:
                skip -= c.sample_count
                continue
            chunks.append((c.erd_path, c.sample_num_start + skip, c.sample_count - skip))
            skip = 0
        out.append(chunks)
    return out


def reference_range(ref_chunks, a, b, tmp_dir):
    pieces, pos = [], 0
    for path, local, count in ref_chunks:
        lo, hi = max(a, pos), min(b, pos + count)
        if lo < hi:
            pieces.append(reference_file_samples(path, local + lo - pos, local + hi - pos, tmp_dir))
        pos += count
    return np.concatenate(pieces, axis=0)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
ALL_KEYS = list(SAMPLES) + ["c0fe2d81"]


@pytest.mark.parametrize("key", ALL_KEYS)
def test_compliance(key):
    r = get_reader(key)
    compliance.header_is_total(r)
    compliance.check_signal_stream_buffer_hierachy(r)
    compliance.count_element(r)
    compliance.read_analogsignals(r)
    compliance.read_events(r)
    compliance.has_annotations(r)


@pytest.mark.parametrize("key", list(SAMPLES))
def test_segments_equal_eegindex_scan(key):
    scan = _import_reference("eegindex.scan")
    r = get_reader(key)
    expected = scan.scan_session(Path(SAMPLE_ROOT) / SAMPLES[key]).segments
    t0 = r.raw_annotations["blocks"][0]["rec_datetime"].timestamp()
    got = [(t0 + r.segment_t_start(0, s), t0 + r.segment_t_stop(0, s)) for s in range(r.segment_count(0))]
    assert len(got) == len(expected)
    np.testing.assert_allclose(np.array(got), np.array(expected), rtol=0, atol=1e-5)
    for s in range(r.segment_count(0)):
        seg_ann = r.raw_annotations["blocks"][0]["segments"][s]
        assert abs(seg_ann["rec_datetime"].timestamp() - expected[s][0]) < 1e-5


@pytest.mark.parametrize("key", list(SAMPLES))
def test_first_samples_equal_reference(key, tmp_path):
    r = get_reader(key)
    ref = reference_segments(Path(SAMPLE_ROOT) / SAMPLES[key])
    stream = seg_stream(r, 0)
    n = min(20000, r.get_signal_size(0, 0, stream))
    got = r.get_analogsignal_chunk(0, 0, 0, n, stream_index=stream)
    expected = reference_range(ref[0], 0, n, tmp_path)
    assert got.dtype == np.int32
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("key", list(SAMPLES))
def test_cross_file_range_equal_reference(key, tmp_path):
    r = get_reader(key)
    ref = reference_segments(Path(SAMPLE_ROOT) / SAMPLES[key])
    for seg_index, chunks in enumerate(ref):
        files = [c[0] for c in chunks]
        if len(set(files)) < 2:
            continue
        pos = 0
        for (path, _, count), (next_path, _, _) in zip(chunks, chunks[1:]):
            pos += count
            if next_path != path:
                break
        stream = seg_stream(r, seg_index)
        a = max(0, pos - 1537)
        b = min(r.get_signal_size(0, seg_index, stream), pos + 1463)
        got = r.get_analogsignal_chunk(0, seg_index, a, b, stream_index=stream, channel_indexes=[0, 3, 7])
        expected = reference_range(chunks, a, b, tmp_path)[:, [0, 3, 7]]
        np.testing.assert_array_equal(got, expected)
        return
    pytest.skip("no segment spans two .erd files in this session")


@pytest.mark.parametrize("key", list(SAMPLES))
def test_mid_file_range_equal_reference(key, tmp_path):
    """A range in the middle of a file, starting half a block after an index point."""
    etc_reader = _import_reference("etc_reader")
    r = get_reader(key)
    ref = reference_segments(Path(SAMPLE_ROOT) / SAMPLES[key])
    # the longest chunk of the longest segment
    seg_index = max(range(len(ref)), key=lambda s: sum(c[2] for c in ref[s]))
    chunk_index = max(range(len(ref[seg_index])), key=lambda i: ref[seg_index][i][2])
    path, local, count = ref[seg_index][chunk_index]
    nums = [e.sample_num for e in etc_reader.read_etc_file(Path(path).with_suffix(".etc"))]
    inside = [n for n in nums if local <= n < local + count]
    j = len(inside) // 2
    start_local = inside[j] + (inside[j + 1] - inside[j]) // 2 if j + 1 < len(inside) else inside[j] + 100
    pos = sum(c[2] for c in ref[seg_index][:chunk_index])
    stream = seg_stream(r, seg_index)
    a = pos + start_local - local
    b = min(a + 2500, r.get_signal_size(0, seg_index, stream))
    got = r.get_analogsignal_chunk(0, seg_index, a, b, stream_index=stream)
    expected = reference_range(ref[seg_index], a, b, tmp_path)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("key", list(SAMPLES))
def test_clips_and_comments_equal_exportnatusevents(key):
    reader_mod = _import_reference("exportnatusevents.reader")
    r = get_reader(key)
    xr = reader_mod.XltekReader(Path(SAMPLE_ROOT) / SAMPLES[key], data_types=reader_mod.CLIP_TYPES | {"Custom"})
    expected = {"clip": [], "comment": []}
    for rec in xr.get_all_records():
        if rec["_datetime_start"] is None:
            continue
        kind = "clip" if rec["_data_type"] in reader_mod.CLIP_TYPES else "comment"
        end = rec["_datetime_end"].timestamp() if rec["_datetime_end"] else None
        expected[kind].append((rec["_datetime_start"].timestamp(), end, str(rec.get("Text", "")).strip()))

    t0 = r.raw_annotations["blocks"][0]["rec_datetime"].timestamp()
    got = {"clip": [], "comment": []}
    for ch, kind in ((0, "clip"), (1, "comment")):
        for s in range(r.segment_count(0)):
            times, durations, labels = r.get_event_timestamps(0, s, ch)
            times = r.rescale_event_timestamp(times, "float64", ch)
            for i in range(times.size):
                start = t0 + times[i]
                end = None
                if durations is not None:
                    d = r.rescale_epoch_duration(durations, "float64", ch)[i]
                    end = start + d if d > 0 else None
                got[kind].append((start, end, str(labels[i])))

    for kind in ("clip", "comment"):
        e = sorted(expected[kind], key=lambda x: (x[0], x[2]))
        g = sorted(got[kind], key=lambda x: (x[0], x[2]))
        assert [x[2] for x in g] == [x[2] for x in e]
        for (gs, ge, _), (es, ee, _) in zip(g, e):
            assert abs(gs - es) < 1e-3
            assert (ge is None) == (ee is None)
            if ge is not None:
                assert abs(ge - ee) < 1e-3


def test_known_values():
    r = get_reader("42a53fa4")
    assert r.segment_count(0) == 24
    gaps = [r.segment_t_start(0, s + 1) - r.segment_t_stop(0, s) for s in range(23)]
    assert sum(g > 1.0 for g in gaps) == 23
    assert sum(r.event_count(0, s, 0) for s in range(24)) == 25
    assert r.get_signal_sampling_rate(0) == 256.0
    assert r.signal_channels_count(0) == 46

    r = get_reader("426dd159")
    assert r.segment_count(0) == 3
    assert r.get_signal_sampling_rate(0) == 512.0
    assert r.signal_channels_count(0) == 276

    r = get_reader("662aafc9")
    assert r.get_signal_sampling_rate(0) == 4096.0
    assert len(r._layout.files) == 2
    assert r.segment_count(0) == 1
    clips = set(r.get_event_timestamps(0, 0, 0)[2])
    assert clips == {"ResearchClip", "ResearchClip2"}
    comments = set(r.get_event_timestamps(0, 0, 1)[2])
    assert comments == {"ResearchComment", "ResearchComment2"}


def test_units_gain_and_names():
    r = get_reader("85ba4df1")
    ch = r.header["signal_channels"]
    assert set(ch["units"]) == {"uV"}
    assert set(ch["dtype"]) == {"int32"}
    hdr = r._layout.files[0].header
    np.testing.assert_allclose(ch["gain"], 8711 / (2**21 - 0.5) * 2**hdr.discard_bits)
    assert np.all(ch["offset"] == 0)
    assert ch["name"][0] == "LA1"  # from the .ent montage
    assert r.segment_t_start(0, 0) == 0.0
    assert r.raw_annotations["blocks"][0]["rec_datetime"].tzinfo is not None


def test_empty_range_shape():
    r = get_reader("42a53fa4")
    x = r.get_analogsignal_chunk(0, 0, 10, 10, stream_index=0)
    assert x.shape == (0, 46)
    x = r.get_analogsignal_chunk(0, 0, 10, 10, stream_index=0, channel_indexes=[1, 2])
    assert x.shape == (0, 2)


def test_rate_change_session():
    """c0fe2d81: 256 Hz and 512 Hz files; a rate change is a segment boundary and a new stream."""
    r = get_reader("c0fe2d81")
    streams = r.header["signal_streams"]
    assert sorted(streams["id"]) == ["256Hz", "512Hz"]
    rates = {sid: r.get_signal_sampling_rate(i) for i, sid in enumerate(streams["id"])}
    assert rates == {"256Hz": 256.0, "512Hz": 512.0}

    n_seg = r.segment_count(0)
    seg_rates = []
    for s in range(n_seg):
        stream = seg_stream(r, s)  # exactly one stream has samples, the other is empty
        other = 1 - stream
        assert r.get_signal_size(0, s, other) == 0
        assert r.get_analogsignal_chunk(0, s, stream_index=other).shape == (0, 46)
        seg_rates.append(rates[streams["id"][stream]])
        assert r.raw_annotations["blocks"][0]["segments"][s]["natus_rate"] == seg_rates[-1]
    # every rate change is a segment boundary of (at most one sample of) zero length
    changes = [s for s in range(n_seg - 1) if seg_rates[s] != seg_rates[s + 1]]
    assert changes
    for s in changes:
        gap = r.segment_t_start(0, s + 1) - r.segment_t_stop(0, s)
        assert abs(gap) <= 1 / min(seg_rates[s], seg_rates[s + 1])
    # the samples of each file are those of the reference decoder
    for s in range(n_seg):
        stream = seg_stream(r, s)
        layout_seg = r._segments[s]
        file_index, local, count = layout_seg.chunks[0]
        n = min(3000, count)
        got = r.get_analogsignal_chunk(0, s, 0, n, stream_index=stream)
        erd_reader = _import_reference("erd_reader")
        path = r._layout.files[file_index].path
        expected = erd_reader.read_erd_samples(path, max_samples=local + n)[:, local:].T
        np.testing.assert_array_equal(got, expected)


def test_neo_io_read_block():
    from neo.io import NatusIO

    if not HAVE_NUMBA:
        pytest.skip("numba not installed")
    d = Path(SAMPLE_ROOT) / SAMPLES["42a53fa4"]
    if not d.is_dir():
        pytest.skip("session folder missing")
    io = NatusIO(dirname=d)
    block = io.read_block(lazy=True)
    assert len(block.segments) == 24
    sig = block.segments[0].analogsignals[0].load(time_slice=None)
    assert sig.shape == (19000, 46)
    assert str(sig.units.dimensionality) == "uV"

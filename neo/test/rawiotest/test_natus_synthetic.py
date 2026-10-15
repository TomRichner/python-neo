"""
NatusRawIO on a small synthetic session: no sample data (nothing is decoded), just
the .erd headers, the .etc indexes, the .stc table and the .snc sync points, enough
for parse_header. Covers the stub the recorder can leave behind: an .erd of a few KB
that stops inside its own header, an .etc of header only, no .stc row. It holds no
samples and must not fail the session.

Also covers decimated monitoring data NeuroWorks can save from outside a recording
(before "Start Recording", or between two): one .etc entry per stored sample while the
stamp steps by 4. It is not part of the recording and is left out (listed in
``NatusLayout.unrecorded``) rather than read as thousands of one-sample segments.
"""

import struct

import pytest

from neo.rawio.natusrawio import NatusRawIO
from neo.rawio.natusrawio.natus_files import GENERIC_HEADER_END, SAMPLE_PACKET_OFFSET, STC_HEADER_END, STC_NAME_BYTES

FILETIME_UNIX_DIFF = 11644473600
STUB_BYTES = 4560
RATE = 256.0
N_CHANNELS = 4
N = 3000  # samples per full file
STEP = 1000  # samples between .etc index points
N_UNREC = 400  # samples per unrecorded file
K = 4  # stamps per unrecorded sample
T0 = 1_700_000_000.0
STEM = "Xx~ Xx_0a1b2c3d-0000-4000-8000-00000000c0de"


def _erd_header(rate=RATE):
    h = bytearray(SAMPLE_PACKET_OFFSET)
    struct.pack_into("<H", h, 16, 9)
    struct.pack_into("<I", h, 20, int(T0))
    struct.pack_into("<d", h, 352, rate)
    struct.pack_into("<i", h, 360, N_CHANNELS)
    struct.pack_into("<i", h, 364, 8)
    struct.pack_into("<i", h, 4464, 1)
    for i in range(N_CHANNELS):
        struct.pack_into("<h", h, 6608 + 2 * i, 32767)
    return bytes(h)


def _etc(entries):
    out = bytearray(GENERIC_HEADER_END)
    struct.pack_into("<h", out, 16, 2)
    for stamp, num in entries:
        out += struct.pack("<iiii", SAMPLE_PACKET_OFFSET + num, stamp, num, 0)
    return bytes(out)


def write_session(parent, kinds, gaps_s=None):
    """kinds: 'full' | 'stub' | 'stub_indexed' | 'unrecorded' per file; gaps_s: seconds before each file."""
    folder = parent / STEM
    folder.mkdir(parents=True)
    gaps_s = gaps_s or [0.0] * len(kinds)
    stc = bytearray(STC_HEADER_END)
    struct.pack_into("<h", stc, 16, 1)
    snc = bytearray(GENERIC_HEADER_END)
    stamp, t = 0, T0
    for i, (kind, gap) in enumerate(zip(kinds, gaps_s)):
        name = f"{STEM}_{i:03d}"
        stamp += int(round(gap * RATE))
        t += gap
        header = _erd_header()
        (folder / f"{name}.erd").write_bytes(header if kind == "full" else header[:STUB_BYTES])
        if kind == "stub":
            (folder / f"{name}.etc").write_bytes(_etc([]))
            continue
        if kind == "unrecorded":
            (folder / f"{name}.etc").write_bytes(_etc([(stamp + K * j, j) for j in range(N_UNREC)]))
            stc += name.encode().ljust(STC_NAME_BYTES, b"\0") + struct.pack(
                "<iiii", stamp, stamp + K * (N_UNREC - 1), 0, N_UNREC
            )
            snc += struct.pack("<IQ", stamp, int(round((t + FILETIME_UNIX_DIFF) * 10_000_000)))
            stamp += K * N_UNREC
            t += K * N_UNREC / RATE
            continue
        (folder / f"{name}.etc").write_bytes(_etc([(stamp + k, k) for k in range(0, N, STEP)]))
        stc += name.encode().ljust(STC_NAME_BYTES, b"\0") + struct.pack("<iiii", stamp, stamp + N - 1, 0, N)
        snc += struct.pack("<IQ", stamp, int(round((t + FILETIME_UNIX_DIFF) * 10_000_000)))
        stamp += N
        t += N / RATE
    (folder / f"{STEM}.stc").write_bytes(stc)
    (folder / f"{STEM}.snc").write_bytes(snc)
    return folder


def _parsed(folder):
    reader = NatusRawIO(str(folder))
    reader.parse_header()
    return reader


def test_stub_is_left_out(tmp_path):
    plain = _parsed(write_session(tmp_path / "plain", ["full", "full", "full"]))
    with_stub = _parsed(write_session(tmp_path / "stub", ["full", "full", "stub", "full"]))
    for reader in (plain, with_stub):
        assert reader.segment_count(0) == 1
        assert reader.get_signal_size(0, 0, 0) == 3 * N
        assert reader.segment_t_start(0, 0) == pytest.approx(0.0)
    assert with_stub._layout.segments[0].t_start_posix == plain._layout.segments[0].t_start_posix == T0


def test_gap_after_stub_stays(tmp_path):
    reader = _parsed(write_session(tmp_path / "gap", ["full", "stub", "full"], [0.0, 0.0, 10.0]))
    assert reader.segment_count(0) == 2
    assert reader.segment_t_start(0, 1) == pytest.approx(N / RATE + 10.0)


def test_indexed_file_with_unreadable_header_is_an_error(tmp_path):
    folder = write_session(tmp_path / "bad", ["full", "stub_indexed", "full"])
    with pytest.raises(ValueError, match=r"_001\.erd is indexed by its \.etc but its header cannot be read"):
        _parsed(folder)


def test_unrecorded_before_and_between_recordings_is_left_out(tmp_path):
    reader = _parsed(write_session(tmp_path / "u", ["unrecorded", "full", "unrecorded", "unrecorded", "full"]))
    assert reader.segment_count(0) == 2
    assert reader.get_signal_size(0, 0, 0) == reader.get_signal_size(0, 1, 0) == N
    segments = reader._layout.segments
    assert segments[0].t_start_posix == pytest.approx(T0 + K * N_UNREC / RATE)
    gap = segments[1].t_start_posix - (segments[0].t_start_posix + N / RATE)
    assert gap == pytest.approx(2 * K * N_UNREC / RATE)
    unrecorded = reader._layout.unrecorded
    assert [(u.path.name[-8:], u.n_samples, u.stamp_step) for u in unrecorded] == [
        ("_000.erd", N_UNREC, K),
        ("_002.erd", N_UNREC, K),
        ("_003.erd", N_UNREC, K),
    ]


def test_session_of_unrecorded_data_only_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="Only unrecorded"):
        _parsed(write_session(tmp_path / "only", ["unrecorded", "unrecorded"]))

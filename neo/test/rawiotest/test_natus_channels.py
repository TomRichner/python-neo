"""
Tests of the Natus headbox channel table and the .erd header checks that need no sample
data (``neo.rawio.natusrawio.natus_channels``, ``natus_files.read_erd_header``).
"""

import logging
import struct

import numpy as np
import pytest

from neo.rawio.natusrawio import natus_files
from neo.rawio.natusrawio.natus_channels import (
    HEADBOXES,
    KINDS,
    UV_PER_COUNT,
    channel_kinds,
    channel_table,
    kind_gain_units,
    physical_names,
)

QUANTUM_AUX = [f"DC{i}" for i in range(1, 17)] + ["TRIG", "OSAT", "PR", "Pleth"]


def test_quantum_kinds_follow_the_tail():
    for n in (276, 148, 84):
        kinds = channel_kinds(20, n)
        assert kinds[: n - 20] == ["eeg"] * (n - 20)
        assert kinds[n - 20 :] == ["dc"] * 16 + ["trigger", "oximetry", "pulse", "pleth"]


def test_quantum_names_from_the_end_of_chan_names():
    names = [f"E{i}" for i in range(1, 257)] + [f"C{i}" for i in range(257, 513)] + QUANTUM_AUX
    assert len(names) == 532
    got = physical_names(20, 276, names)
    assert got[:256] == names[:256]
    assert got[256:] == QUANTUM_AUX


def test_quantum_names_one_per_channel_are_kept_in_order():
    names = [f"E{i}" for i in range(1, 129)] + QUANTUM_AUX
    assert physical_names(20, 148, names) == names


def test_headbox_17_layout():
    kinds = channel_kinds(17, 46)
    assert kinds[:40] == ["eeg"] * 40
    assert kinds[40:] == ["dc"] * 4 + ["oximetry", "pulse"]
    names = [f"E{i}" for i in range(1, 41)] + ["DC1", "DC2", "DC3", "DC4", "O2Sat", "HR"]
    assert physical_names(17, 46, names) == names


def test_longer_list_without_tail_is_read_in_order():
    # headbox 3: 256 names (OSAT and PR at 129-130), 128 physical channels, all EEG
    names = [f"E{i}" for i in range(1, 129)] + ["OSAT", "PR"] + [f"C{i}" for i in range(131, 257)]
    assert physical_names(3, 128, names) == names[:128]
    assert set(channel_kinds(3, 128)) == {"eeg"}


def test_gains_and_units():
    d = 6
    assert kind_gain_units("eeg", 17, d) == (pytest.approx(8711 / (2**21 - 0.5) * 2**d), "uV")
    assert kind_gain_units("dc", 17, d) == (pytest.approx(10_800_000 / 65536 / 2**6 * 2**d / 1000), "mV")
    assert kind_gain_units("dc", 20, d) == (pytest.approx(10_800_000 / 65536 / 2**6 * 2**d / 1000), "mV")
    assert kind_gain_units("dc", 15, d) == (pytest.approx(10_000_000 / 65536 / 2**6 * 2**d / 1000), "mV")
    assert kind_gain_units("dc", 4, d) == (pytest.approx(5_000_000 / (2**10 - 0.5) / 2**6 * 2**d / 1000), "mV")
    assert kind_gain_units("oximetry", 17, d) == (pytest.approx(1.0), "%")
    assert kind_gain_units("pulse", 17, d) == (pytest.approx(1.0), "bpm")
    assert kind_gain_units("pleth", 20, d)[1] == "mV"
    assert kind_gain_units("trigger", 20, d) == (1.0, "")


def test_wonambi_layouts():
    """Non-EEG positions as in Wonambi's ``ktlx._calculate_conversion``."""
    expected = {
        4: {24: "dc", 27: "dc"},
        6: {32: "dc", 35: "dc"},
        8: {25: "oximetry", 26: "pulse"},
        9: {33: "oximetry", 34: "pulse"},
        14: {38: "dc", 47: "dc", 48: "oximetry", 49: "pulse"},
        15: {27: "eeg", 28: "dc", 31: "dc", 32: "oximetry", 33: "pulse"},
        17: {39: "eeg", 40: "dc", 43: "dc", 44: "oximetry", 45: "pulse"},
        21: {127: "eeg", 128: "oximetry", 129: "pulse", 130: "eeg"},
        22: {32: "dc", 39: "dc", 40: "oximetry", 41: "pulse", 42: "dc"},
        23: {32: "dc", 35: "dc", 36: "oximetry", 37: "pulse", 38: "dc"},
    }
    for headbox, positions in expected.items():
        kinds = channel_kinds(headbox, 256)
        for i, kind in positions.items():
            assert kinds[i] == kind, (headbox, i)
    for headbox in (1, 3, 19):
        assert set(channel_kinds(headbox, 64)) == {"eeg"}
    for layout in HEADBOXES.values():
        assert set(layout.tail) <= set(KINDS)
        for _, block in layout.blocks:
            assert set(block) <= set(KINDS)


def test_unknown_headbox_is_all_eeg_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        table = channel_table(99, 8, 6, None)
    assert [t[1] for t in table] == ["eeg"] * 8
    assert all(t[3] == "uV" and t[2] == pytest.approx(UV_PER_COUNT * 64) for t in table)
    assert "headbox type 99" in caplog.text


def _erd_header(path, schema=9, n=4, freq=None, shorted=None):
    h = bytearray(natus_files.SAMPLE_PACKET_OFFSET)
    struct.pack_into("<H", h, natus_files.ERD_FILE_SCHEMA_OFFSET, schema)
    struct.pack_into("<d", h, natus_files.ERD_SAMP_FREQ_OFFSET, 256.0)
    struct.pack_into("<i", h, natus_files.ERD_NUM_CHANS_OFFSET, n)
    struct.pack_into("<i", h, natus_files.ERD_HEADBOX_TYPE_ARRAY_OFFSET, 17)
    struct.pack_into("<i", h, natus_files.ERD_DISCARD_BITS_OFFSET, 6)
    struct.pack_into(f"<{n}h", h, natus_files.ERD_SHORTED_CHANS_OFFSET, *(shorted or [0] * n))
    struct.pack_into(f"<{n}h", h, natus_files.ERD_FREQUENCY_FACTOR_OFFSET, *(freq or [32767] * n))
    path.write_bytes(bytes(h))
    return path


@pytest.mark.parametrize("schema", [1, 3, 7, 10])
def test_unsupported_erd_schema_is_refused(tmp_path, schema):
    path = _erd_header(tmp_path / "x.erd", schema=schema)
    with pytest.raises(ValueError, match="Unsupported ERD schema"):
        natus_files.read_erd_header(path)


def test_erd_header_fields(tmp_path, caplog):
    path = _erd_header(tmp_path / "x.erd", schema=8)
    with caplog.at_level(logging.WARNING):
        h = natus_files.read_erd_header(path)
    assert h.headbox_type == 17
    assert h.frequency_factor == (32767,) * 4
    assert "frequency_factor" not in caplog.text


def test_frequency_factor_other_than_full_rate_warns(tmp_path, caplog):
    path = _erd_header(tmp_path / "x.erd", freq=[32767, 16383, 32767, 100], shorted=[0, 0, 0, 1])
    with caplog.at_level(logging.WARNING):
        natus_files.read_erd_header(path)
    assert "frequency_factor is not 32767 for 1 channel(s)" in caplog.text
    assert "physical channel 2" in caplog.text


def test_channel_table_matches_kind_gain_units():
    table = channel_table(17, 46, 6, None)
    assert [t[0] for t in table] == [None] * 46
    for name, kind, gain, units in table:
        assert (gain, units) == kind_gain_units(kind, 17, 6)
    assert np.isclose(table[40][2], 0.16479492, atol=1e-8)

"""Synthetic real-packet EOF recovery: preserve samples, time and strict errors."""
import struct

import numpy as np
import pytest

from neo.rawio.natusrawio import NatusRawIO
from neo.rawio.natusrawio.natus_files import GENERIC_HEADER_END, SAMPLE_PACKET_OFFSET, STC_HEADER_END, STC_NAME_BYTES
from neo.test.rawiotest.test_natus_synthetic import _erd_header, FILETIME_UNIX_DIFF, RATE, T0

GUID = '00000000-0000-4000-8000-000000000023'


def packets(start, count):
    if count == 0:
        return b''
    values = [100+start+10*c for c in range(4)]
    return b'\x00\x0f' + b'\xff\xff'*4 + struct.pack('<iiii', *values) + b'\x00\x00\x01\x01\x01\x01'*(count-1)


def make_session(parent, mode='partial', n=8, block=4, files=3):
    root = parent / ('Synthetic_' + GUID)
    root.mkdir(parents=True)
    stc = bytearray(STC_HEADER_END)
    struct.pack_into('<h', stc, 16, 1)
    snc = bytearray(GENERIC_HEADER_END)
    for fi in range(files):
        stem = 'Synthetic_' + GUID + f'_{fi:03d}'
        payload = bytearray()
        etc = bytearray(GENERIC_HEADER_END)
        struct.pack_into('<h', etc, 16, 2)
        for a in range(0, n, block):
            count = min(block, n-a)
            struct_tail = fi == 1 and a+count == n
            body = packets(fi*n+a, count)
            if struct_tail and mode in ('partial', 'empty'):
                body = packets(fi*n+a, 2) + b'\x00\x00\x01' if mode == 'partial' else b''
            if fi == 1 and a == 0 and mode == 'interior':
                body = packets(fi*n+a, 2) + b'\x00\x00\x01'
            etc += struct.pack('<iiii', SAMPLE_PACKET_OFFSET+len(payload), fi*n+a, a, count)
            payload += body
        (root / (stem+'.erd')).write_bytes(_erd_header() + payload)
        (root / (stem+'.etc')).write_bytes(etc)
        stc += stem.encode().ljust(STC_NAME_BYTES, b'\x00') + struct.pack('<iiii', fi*n, (fi+1)*n-1, fi*n, n)
        snc += struct.pack('<IQ', fi*n, int(round((T0+fi*n/RATE+FILETIME_UNIX_DIFF)*10_000_000)))
    (root / 'Synthetic.stc').write_bytes(stc)
    (root / 'Synthetic.snc').write_bytes(snc)
    return root


def read_all(reader):
    return [reader.get_analogsignal_chunk(block_index=0,seg_index=i, stream_index=0)
            for i in range(reader.segment_count(0))]


def test_partial_tail_becomes_gap_with_exact_values_and_clock(tmp_path):
    root = make_session(tmp_path)
    reader = NatusRawIO(root, recover_truncated_tails=True)
    reader.parse_header()
    assert reader.segment_count(0) == 2
    values = read_all(reader)
    expected = lambda samples: 100+np.asarray(samples)[:, None]+np.arange(4)[None, :]*10
    np.testing.assert_array_equal(values[0], expected(np.arange(14)))
    np.testing.assert_array_equal(values[1], expected(np.arange(16, 24)))
    assert reader.segment_t_start(block_index=0,seg_index=1) == 16/RATE
    assert reader.segment_t_stop(block_index=0,seg_index=0) == 14/RATE
    gap = reader._layout.tail_gaps[0]
    assert gap['sample_count'] == 2 and gap['complete_tail_samples'] == 2
    assert gap['start_stamp'] == 14 and gap['stop_stamp'] == 16
    assert gap['start_posix'] == T0+14/RATE
    assert reader._layout.recover_truncated_tails() == [gap]


def test_opt_in_leaves_legacy_layout_unchanged(tmp_path):
    root = make_session(tmp_path)
    reader = NatusRawIO(root)
    reader.parse_header()
    assert reader.segment_count(0) == 1 and reader.get_signal_size(0,0,0) == 24
    assert not hasattr(reader._layout, 'tail_gaps')


@pytest.mark.parametrize('mode,missing', [('intact', 0), ('empty', 4)])
def test_intact_or_empty_final_index_block(tmp_path, mode, missing):
    reader = NatusRawIO(make_session(tmp_path, mode=mode), recover_truncated_tails=True)
    reader.parse_header()
    assert sum(len(v) for v in read_all(reader)) == 24-missing
    assert sum(g['sample_count'] for g in reader._layout.tail_gaps) == missing


def test_interior_corruption_remains_a_strict_error(tmp_path):
    reader = NatusRawIO(make_session(tmp_path, mode='interior'), recover_truncated_tails=True)
    reader.parse_header()
    assert not reader._layout.tail_gaps
    with pytest.raises(ValueError, match='strict mode prohibits padding'):
        read_all(reader)


def test_bounded_decoder_carries_state_across_tail_batches(tmp_path):
    reader = NatusRawIO(make_session(tmp_path, mode='intact', n=9000, block=9000, files=1),
                        recover_truncated_tails=True)
    reader.parse_header()
    assert not reader._layout.tail_gaps
    values = read_all(reader)[0]
    np.testing.assert_array_equal(values[:, 0], 100+np.arange(9000))


def test_whole_indexed_file_missing_preserves_both_neighbors(tmp_path):
    reader=NatusRawIO(make_session(tmp_path,mode='empty',n=8,block=8),recover_truncated_tails=True)
    reader.parse_header()
    assert [len(v) for v in read_all(reader)]==[8,8]
    assert reader._layout.tail_gaps[0]['sample_count']==8
    assert reader.segment_t_start(0,1)==16/RATE


def test_unindexed_stub_is_still_ignored(tmp_path):
    root=make_session(tmp_path,mode='intact')
    (root/'Unused.erd').write_bytes(_erd_header()[:4560])
    etc=bytearray(GENERIC_HEADER_END)
    struct.pack_into('<h',etc,16,2)
    (root/'Unused.etc').write_bytes(etc)
    reader=NatusRawIO(root,recover_truncated_tails=True)
    reader.parse_header()
    assert not reader._layout.tail_gaps
    assert sum(len(v) for v in read_all(reader))==24


@pytest.mark.parametrize('explicit_span',[True,False])
def test_tail_recovery_requires_corroborated_extent_without_stc(tmp_path,explicit_span):
    root=make_session(tmp_path)
    (root/'Synthetic.stc').unlink()
    if not explicit_span:
        path=next(root.glob('*_001.etc'))
        data=bytearray(path.read_bytes())
        struct.pack_into('<i',data,len(data)-4,0)
        path.write_bytes(data)
    reader=NatusRawIO(root,recover_truncated_tails=True)
    if explicit_span:
        reader.parse_header()
        assert reader._layout.tail_gaps[0]['extent_evidence']=='ETC_sample_span'
    else:
        with pytest.raises(ValueError,match='corroborated indexed extent'):
            reader.parse_header()


def test_invalid_final_offset_is_not_reclassified_as_gap(tmp_path):
    root=make_session(tmp_path)
    path=next(root.glob('*_001.etc'))
    data=bytearray(path.read_bytes())
    erd=next(root.glob('*_001.erd'))
    struct.pack_into('<i',data,len(data)-16,erd.stat().st_size+100)
    path.write_bytes(data)
    with pytest.raises(ValueError,match='outside the ERD payload'):
        NatusRawIO(root,recover_truncated_tails=True).parse_header()

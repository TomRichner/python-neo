"""
Decoder for the Natus/XLTEK ``.erd`` sample stream (needs numba).

Packet layout, as read by the reference decoder
(``xlteknatustomat/erd_reader.read_erd_samples``):

  event byte (skipped)
  + delta mask: ceil(num_channels / 8) bytes, one bit per *physical* channel
    (bits of shorted channels are present but skipped)
  + one delta per recorded channel, in channel order:
      mask bit 0 -> 1-byte signed delta
      mask bit 1 -> 2-byte signed little-endian delta; 0xFFFF is an escape: the
                    channel's absolute value follows as a 4-byte signed int after
                    all deltas of the packet (in channel order)

value[ch] = running int32 sum of the deltas, replaced by the absolute value on an
escape. At every ``.etc`` index point all channels are written as absolute values,
so decoding can start there with zero state and is exact from the first sample.

Truncated tail packets are handled exactly like the reference: a packet whose
delta mask does not fit ends decoding; a packet whose deltas run off the end is
still emitted with the channels decoded so far.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False


def _decode_packets(buf, pos, mask_len, mask_byte, mask_shift, chan_sums, overflow, start, count):
    """Decode packets from buf (uint8, a piece of the sample stream).

    buf         uint8 array; byte ``pos`` is the event byte of a packet
    pos         byte position to start at (a packet boundary)
    mask_len    delta mask length in bytes
    mask_byte   int32[n_rec]: byte index in the mask of each recorded channel's bit
    mask_shift  int32[n_rec]: bit index within that byte
    chan_sums   int32[n_rec]: running values (state, updated in place)
    overflow    uint8[n_rec]: pending-escape flags (state, updated in place)
    start       number of packets to decode without storing (skip)
    count       number of packets to store after the skipped ones

    Returns (out[count, n_rec] int32, n_stored, end_pos).
    """
    n_rec = chan_sums.shape[0]
    data_len = buf.shape[0]
    out = np.empty((count, n_rec), dtype=np.int32)
    # Worst-case packet size: every channel a 2-byte escape plus a 4-byte value.
    max_packet = 1 + mask_len + 6 * n_rec
    total = start + count
    k = 0
    n_out = 0
    while k < total:
        if pos >= data_len:
            break
        pos += 1  # event byte
        if pos >= data_len:
            break
        if pos + mask_len > data_len:
            break
        mpos = pos
        pos += mask_len
        any_over = False
        if pos - 1 - mask_len + max_packet <= data_len:
            # Fast path: whole packet surely in the buffer.
            for i in range(n_rec):
                bit = (buf[mpos + mask_byte[i]] >> mask_shift[i]) & 1
                if bit:
                    v = np.int32(buf[pos]) | (np.int32(buf[pos + 1]) << 8)
                    pos += 2
                    if v == 0xFFFF:
                        overflow[i] = 1
                        any_over = True
                    else:
                        if v >= 0x8000:
                            v -= 0x10000
                        chan_sums[i] += v
                else:
                    b = np.int32(buf[pos])
                    pos += 1
                    if b >= 128:
                        b -= 256
                    chan_sums[i] += b
            if any_over:
                for i in range(n_rec):
                    if overflow[i]:
                        u = (
                            np.uint32(buf[pos])
                            | (np.uint32(buf[pos + 1]) << 8)
                            | (np.uint32(buf[pos + 2]) << 16)
                            | (np.uint32(buf[pos + 3]) << 24)
                        )
                        chan_sums[i] = np.int32(u)
                        pos += 4
                        overflow[i] = 0
        else:
            # Slow path near the end of the buffer: same bounds checks as the reference.
            for i in range(n_rec):
                bit = (buf[mpos + mask_byte[i]] >> mask_shift[i]) & 1
                if bit:
                    if pos + 2 > data_len:
                        break
                    v = np.int32(buf[pos]) | (np.int32(buf[pos + 1]) << 8)
                    pos += 2
                    if v == 0xFFFF:
                        overflow[i] = 1
                        any_over = True
                    else:
                        if v >= 0x8000:
                            v -= 0x10000
                        chan_sums[i] += v
                else:
                    if pos >= data_len:
                        break
                    b = np.int32(buf[pos])
                    pos += 1
                    if b >= 128:
                        b -= 256
                    chan_sums[i] += b
            if any_over:
                for i in range(n_rec):
                    if overflow[i]:
                        if pos + 4 > data_len:
                            break
                        u = (
                            np.uint32(buf[pos])
                            | (np.uint32(buf[pos + 1]) << 8)
                            | (np.uint32(buf[pos + 2]) << 16)
                            | (np.uint32(buf[pos + 3]) << 24)
                        )
                        chan_sums[i] = np.int32(u)
                        pos += 4
                        overflow[i] = 0
        if k >= start:
            for i in range(n_rec):
                out[n_out, i] = chan_sums[i]
            n_out += 1
        k += 1
    return out, n_out, pos


if HAVE_NUMBA:
    decode_packets = njit(nogil=True, cache=True, boundscheck=False)(_decode_packets)
else:  # pragma: no cover
    decode_packets = None


class ErdLayout:
    """Per-file constants the decoder needs (mask length and bit positions)."""

    def __init__(self, num_channels: int, rec_to_phys_map):
        self.mask_len = math.ceil(num_channels / 8)
        bits = np.asarray(rec_to_phys_map, dtype=np.int32) - 1
        self.mask_byte = (bits // 8).astype(np.int32)
        self.mask_shift = (bits % 8).astype(np.int32)
        self.n_rec = bits.size


def decode_range(buf: np.ndarray, layout: ErdLayout, skip: int, count: int) -> np.ndarray:
    """Decode ``count`` samples after skipping ``skip``, starting at buf[0] with zero state.

    ``buf`` must start at an ``.etc`` index point (all channels absolute there).
    Returns int32 (n, n_rec) with n <= count (fewer if the buffer ends first).
    """
    if decode_packets is None:  # pragma: no cover
        raise ImportError("NatusRawIO needs numba: pip install neo[natus]")
    chan_sums = np.zeros(layout.n_rec, dtype=np.int32)
    overflow = np.zeros(layout.n_rec, dtype=np.uint8)
    out, n, _ = decode_packets(
        buf, 0, layout.mask_len, layout.mask_byte, layout.mask_shift, chan_sums, overflow, skip, count
    )
    return out[:n]

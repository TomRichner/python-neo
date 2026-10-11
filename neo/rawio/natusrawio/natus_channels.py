"""
Physical channels of a Natus/XLTEK headbox: their kind, gain, units and name.

The ``.erd`` header gives the headbox type (offset 4464), the number of physical
channels and the discard bits; samples are stored in physical channel order. The
``.ent`` ``ChanNames`` list names the inputs of the headbox layout, which is not
always one name per physical channel:

* Most headboxes: one name per physical channel, in order (headbox 17: 46 names for
  46 channels, DC1-DC4, O2Sat and HR last).
* Headbox 20 (Quantum): 532 names, i.e. 512 amplifier inputs then DC1-DC16, TRIG,
  OSAT, PR and Pleth, for 276 physical channels: 256 amplifier inputs then the same 20
  auxiliary channels. The auxiliary channels are therefore named from the tail of the
  list, not from ``ChanNames[phys - 1]``.
* When the list is longer than the physical channels and the headbox keeps its
  auxiliary block elsewhere (headbox 3: 256 names, OSAT and PR at 129-130, 128
  physical channels, all amplifier inputs), the first names are used in order.

Gains (physical units per stored count, before the factor ``2**discard_bits``), by kind:

* ``eeg``: 8711 / (2**21 - 0.5) uV, units uV (xlteknatustomat, Wonambi ``ktlx``).
* ``dc``, headboxes 4 and 6: 5e6 / (2**10 - 0.5) / 2**6 uV, units mV (Wonambi
  ``_calculate_conversion``).
* ``dc``, headboxes 14, 17, 22 and 23: 10.8e6 / 65536 / 2**6 uV, units mV (Wonambi).
* ``dc``, headbox 15: 10e6 / 65536 / 2**6 uV, units mV (Wonambi).
* ``dc``, headbox 20 (Quantum): 10.8e6 / 65536 / 2**6 uV, units mV. Assumed as for 17,
  22 and 23; Wonambi has no headbox 20.
* ``oximetry`` (SpO2, OSAT, O2Sat): 1 / 2**6, units % (Wonambi).
* ``pulse`` (PR, HR): 1 / 2**6, units bpm (Wonambi).
* ``pleth`` (Quantum): the DC gain, units mV. Assumed: Wonambi gives the one channel
  after the oximetry pair of headboxes 22 and 23 the DC gain (it is read here as
  ``dc``, since Wonambi does not name it).
* ``trigger`` (Quantum TRIG): 1, no units; a bit field kept as stored. No source.

Wonambi gives every factor in uV; DC and pleth channels are given here in mV (the
Wonambi factor / 1000). No sample session holds a connected DC, oximetry or pleth
input, so the DC, oximetry and pleth gains could not be checked against known values:
oximetry and pulse inputs without a sensor read 131070 (or 0), the Quantum TRIG input
reads 511 and its Pleth input 0.

Headbox types not in the table are read with the EEG gain on every channel (a warning
is logged).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

KINDS = ("eeg", "dc", "trigger", "oximetry", "pulse", "pleth")

UV_PER_COUNT = 8711 / (2**21 - 0.5)
_DC_10_8 = 10_800_000 / 65536 / 2**6  # uV per count
_DC_10_0 = 10_000_000 / 65536 / 2**6
_DC_OLD = 5_000_000 / (2**10 - 0.5) / 2**6
_OXI = 1 / 2**6


@dataclass(frozen=True)
class HeadboxLayout:
    """Non-EEG blocks of a headbox's physical channels.

    ``blocks``: (first physical channel, 0-based, kinds) for layouts fixed from the
    start; ``tail``: kinds of an auxiliary block at the end of the physical channels,
    whose names are the last ones of the ``ChanNames`` list (Quantum).
    """

    name: str
    dc_uv: float = _DC_10_8
    blocks: tuple[tuple[int, tuple[str, ...]], ...] = ()
    tail: tuple[str, ...] = ()


_OXI_PAIR = ("oximetry", "pulse")

HEADBOXES: dict[int, HeadboxLayout] = {
    1: HeadboxLayout("type 1"),
    3: HeadboxLayout("type 3"),
    4: HeadboxLayout("type 4", dc_uv=_DC_OLD, blocks=((24, ("dc",) * 4),)),
    6: HeadboxLayout("type 6", dc_uv=_DC_OLD, blocks=((32, ("dc",) * 4),)),
    8: HeadboxLayout("type 8", blocks=((25, _OXI_PAIR),)),
    9: HeadboxLayout("type 9", blocks=((33, _OXI_PAIR),)),
    14: HeadboxLayout("type 14", blocks=((38, ("dc",) * 10), (48, _OXI_PAIR))),
    15: HeadboxLayout("type 15", dc_uv=_DC_10_0, blocks=((28, ("dc",) * 4), (32, _OXI_PAIR))),
    17: HeadboxLayout("type 17", blocks=((40, ("dc",) * 4), (44, _OXI_PAIR))),
    19: HeadboxLayout("type 19"),
    20: HeadboxLayout("Quantum", tail=("dc",) * 16 + ("trigger", "oximetry", "pulse", "pleth")),
    21: HeadboxLayout("type 21", blocks=((128, _OXI_PAIR),)),
    # Wonambi gives channel 43 of headboxes 22 and 23 the DC gain without naming it
    22: HeadboxLayout("type 22", blocks=((32, ("dc",) * 8), (40, _OXI_PAIR), (42, ("dc",)))),
    23: HeadboxLayout("type 23", blocks=((32, ("dc",) * 4), (36, _OXI_PAIR), (38, ("dc",)))),
}


def channel_kinds(headbox_type: int, num_channels: int) -> list[str]:
    """Kind of each physical channel (index 0 = physical channel 1)."""
    kinds = ["eeg"] * num_channels
    layout = HEADBOXES.get(headbox_type)
    if layout is None:
        return kinds
    for first, block in layout.blocks:
        for j, kind in enumerate(block):
            if first + j < num_channels:
                kinds[first + j] = kind
    if layout.tail and num_channels > len(layout.tail):
        kinds[num_channels - len(layout.tail) :] = layout.tail
    return kinds


def kind_gain_units(kind: str, headbox_type: int, discard_bits: int) -> tuple[float, str]:
    """(gain, units) of a channel of this kind: physical value = stored count * gain."""
    layout = HEADBOXES.get(headbox_type)
    dc_uv = layout.dc_uv if layout is not None else _DC_10_8
    shift = 2**discard_bits
    if kind in ("dc", "pleth"):
        return dc_uv * shift / 1000.0, "mV"
    if kind == "oximetry":
        return _OXI * shift, "%"
    if kind == "pulse":
        return _OXI * shift, "bpm"
    if kind == "trigger":
        return 1.0, ""
    return UV_PER_COUNT * shift, "uV"


def physical_names(headbox_type: int, num_channels: int, names: list[str] | None) -> list[str | None]:
    """Name of each physical channel from the ``.ent`` ``ChanNames`` list (None if none).

    One name per channel in order, except for a headbox whose auxiliary block is named
    from the end of a longer list (Quantum)."""
    if not names:
        return [None] * num_channels
    out: list[str | None] = [names[i] if i < len(names) else None for i in range(num_channels)]
    layout = HEADBOXES.get(headbox_type)
    if layout is not None and layout.tail and len(names) != num_channels:
        n_aux = len(layout.tail)
        n_amp = num_channels - n_aux
        if n_amp > 0 and len(names) >= num_channels:
            for j in range(n_aux):
                out[n_amp + j] = names[len(names) - n_aux + j]
        else:
            logger.warning(
                "ChanNames has %d names for %d physical channels of a %s headbox; "
                "naming the channels in order",
                len(names),
                num_channels,
                layout.name,
            )
    return out


def channel_table(headbox_type: int, num_channels: int, discard_bits: int, names: list[str] | None):
    """Per physical channel: (name or None, kind, gain, units)."""
    if headbox_type not in HEADBOXES:
        logger.warning(
            "Natus headbox type %d is not in the channel table; every channel is read as EEG (uV)", headbox_type
        )
    kinds = channel_kinds(headbox_type, num_channels)
    pnames = physical_names(headbox_type, num_channels, names)
    out = []
    for i in range(num_channels):
        gain, units = kind_gain_units(kinds[i], headbox_type, discard_bits)
        out.append((pnames[i], kinds[i], gain, units))
    return out

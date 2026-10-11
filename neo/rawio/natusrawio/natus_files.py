"""
Small readers for the files of a Natus/XLTEK (NeuroWorks) session folder.

Ported from the ``xlteknatustomat`` readers (``erd_reader``, ``etc_reader``,
``stc_reader``, ``snc_reader``, ``montage_extractor``) and from the NtKeys parser
of ``exportnatusevents.reader``. Those packages stay the reference; nothing here
imports them.

Every file starts with a 352-byte generic header.

* ``.erd``: raw data. File-specific header up to byte 8656, then the
  delta-compressed sample stream (see ``erd_decode``).
* ``.etc``: index of one ``.erd``: 16-byte entries (byte offset into the
  ``.erd``, global samplestamp, local sample number, sample span). The sample
  span field is unreliable and is not used.
* ``.stc``: one 272-byte entry per ``.erd`` (name, first and last samplestamp,
  sample number, number of stored samples), after 56 more header bytes.
* ``.snc``: 12-byte entries (samplestamp, Windows FILETIME) for wall-clock time.
* ``.ent``: NtKeys records (notes, clips, montages...).
"""

from __future__ import annotations

import logging
import os
import re
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

GENERIC_HEADER_END = 352

# .erd header offsets
ERD_FILE_SCHEMA_OFFSET = 16
ERD_CREATION_TIME_OFFSET = 20
ERD_SAMP_FREQ_OFFSET = 352
ERD_NUM_CHANS_OFFSET = 360
ERD_DELTA_BITS_OFFSET = 364
ERD_HEADBOX_TYPE_ARRAY_OFFSET = 4464
ERD_DISCARD_BITS_OFFSET = 4556
ERD_SHORTED_CHANS_OFFSET = 4560
ERD_FREQUENCY_FACTOR_OFFSET = 6608
SAMPLE_PACKET_OFFSET = 8656
# Schemas 8 and 9 share the packet layout decoded here (delta mask, 0xFFFF escape);
# schema 7 has a 1-byte escape and no delta mask (Wonambi ktlx notes).
SUPPORTED_ERD_SCHEMAS = {8, 9}
# frequency_factor of a channel recorded at the file's sample rate; any other value
# means a channel at a different rate, with a frequency byte in each packet (not decoded).
FULL_RATE_FREQUENCY_FACTOR = 32767

# .etc
ETC_SCHEMA_OFFSET = 16
ETC_ENTRY_SIZE = 16
SUPPORTED_ETC_SCHEMAS = {2, 3}

# .stc
STC_SCHEMA_OFFSET = 16
STC_HEADER_END = GENERIC_HEADER_END + 56
STC_NAME_BYTES = 256
STC_ENTRY_SIZE = STC_NAME_BYTES + 16
SUPPORTED_STC_SCHEMAS = {1}

# .snc
SNC_ENTRY_SIZE = 12
FILETIME_UNIX_DIFF = 11644473600  # seconds from 1601-01-01 to 1970-01-01

# .ent
CLIP_TYPES = {"ClipNote", "ClipNoteComment"}
COMMENT_TYPE = "Custom"

# Microvolts per count before the discard bits are applied.
UV_PER_COUNT = 8711 / (2**21 - 0.5)


# ----------------------------------------------------------------------------
# .erd header
# ----------------------------------------------------------------------------
@dataclass
class ErdHeader:
    file_schema: int
    creation_time: datetime | None
    sample_freq: float
    num_channels: int  # physical channels (bits in the delta mask)
    delta_bits: int
    discard_bits: int
    rec_to_phys_map: list[int]  # 1-based physical channel of each recorded channel
    headbox_type: int = 0  # first of the four headbox types (offset 4464)
    frequency_factor: tuple[int, ...] = ()  # per physical channel (offset 6608)

    @property
    def num_recorded_chans(self) -> int:
        return len(self.rec_to_phys_map)

    @property
    def gain(self) -> float:
        """Microvolts per count."""
        return UV_PER_COUNT * 2**self.discard_bits


def read_erd_header(erd_path) -> ErdHeader:
    with open(erd_path, "rb") as f:
        h = f.read(SAMPLE_PACKET_OFFSET)
    if len(h) < SAMPLE_PACKET_OFFSET:
        raise ValueError(f"ERD file too small for its header: {erd_path}")
    file_schema = struct.unpack_from("<H", h, ERD_FILE_SCHEMA_OFFSET)[0]
    if file_schema not in SUPPORTED_ERD_SCHEMAS:
        raise ValueError(
            f"Unsupported ERD schema {file_schema} in {erd_path} (supported: {sorted(SUPPORTED_ERD_SCHEMAS)})"
        )
    creation_raw = struct.unpack_from("<I", h, ERD_CREATION_TIME_OFFSET)[0]
    try:
        creation_time = datetime.fromtimestamp(creation_raw, tz=timezone.utc)
    except (ValueError, OSError, OverflowError):
        creation_time = None
    sample_freq = struct.unpack_from("<d", h, ERD_SAMP_FREQ_OFFSET)[0]
    num_channels = struct.unpack_from("<i", h, ERD_NUM_CHANS_OFFSET)[0]
    delta_bits = struct.unpack_from("<i", h, ERD_DELTA_BITS_OFFSET)[0]
    discard_bits = struct.unpack_from("<i", h, ERD_DISCARD_BITS_OFFSET)[0]
    n_short = min(num_channels, 1024)
    shorted = struct.unpack_from(f"<{n_short}h", h, ERD_SHORTED_CHANS_OFFSET)
    rec_to_phys_map = [i + 1 for i in range(n_short) if shorted[i] == 0]
    headbox_type = struct.unpack_from("<i", h, ERD_HEADBOX_TYPE_ARRAY_OFFSET)[0]
    frequency_factor = struct.unpack_from(f"<{n_short}h", h, ERD_FREQUENCY_FACTOR_OFFSET)
    odd = [i + 1 for i in range(n_short) if shorted[i] == 0 and frequency_factor[i] != FULL_RATE_FREQUENCY_FACTOR]
    if odd:
        logger.warning(
            "%s: frequency_factor is not %d for %d channel(s) (first: physical channel %d, value %d); "
            "channels at another rate are not supported and their samples may be wrong",
            Path(erd_path).name,
            FULL_RATE_FREQUENCY_FACTOR,
            len(odd),
            odd[0],
            frequency_factor[odd[0] - 1],
        )
    return ErdHeader(
        file_schema=file_schema,
        creation_time=creation_time,
        sample_freq=sample_freq,
        num_channels=num_channels,
        delta_bits=delta_bits,
        discard_bits=discard_bits,
        rec_to_phys_map=rec_to_phys_map,
        headbox_type=headbox_type,
        frequency_factor=tuple(frequency_factor),
    )


# ----------------------------------------------------------------------------
# .etc
# ----------------------------------------------------------------------------
def read_etc_file(etc_path) -> np.ndarray:
    """Return an int64 array (n, 4): offset, samplestamp, sample_num, sample_span."""
    with open(etc_path, "rb") as f:
        raw = f.read()
    if len(raw) < GENERIC_HEADER_END:
        raise ValueError(f"ETC file too small: {etc_path}")
    schema = struct.unpack_from("<h", raw, ETC_SCHEMA_OFFSET)[0]
    if schema not in SUPPORTED_ETC_SCHEMAS:
        raise ValueError(f"Unsupported ETC schema {schema} in {etc_path}")
    n = (len(raw) - GENERIC_HEADER_END) // ETC_ENTRY_SIZE
    arr = np.frombuffer(raw, dtype="<i4", count=n * 4, offset=GENERIC_HEADER_END)
    return arr.reshape(n, 4).astype(np.int64)


# ----------------------------------------------------------------------------
# .stc
# ----------------------------------------------------------------------------
@dataclass
class StcEntry:
    segment_name: str
    start_stamp: int
    end_stamp: int
    sample_num: int
    sample_span: int


def read_stc_file(stc_path) -> list[StcEntry]:
    with open(stc_path, "rb") as f:
        data = f.read()
    if len(data) < STC_HEADER_END:
        raise ValueError(f"STC file too small: {stc_path}")
    schema = struct.unpack_from("<h", data, STC_SCHEMA_OFFSET)[0]
    if schema not in SUPPORTED_STC_SCHEMAS:
        raise ValueError(f"Unsupported STC schema {schema} in {stc_path}")
    body = data[STC_HEADER_END:]
    if len(body) % STC_ENTRY_SIZE != 0:
        raise ValueError(f"Unexpected STC size in {stc_path}")
    entries = []
    for i in range(len(body) // STC_ENTRY_SIZE):
        pos = i * STC_ENTRY_SIZE
        name = body[pos : pos + STC_NAME_BYTES].split(b"\x00")[0].decode("latin-1")
        start, end, num, span = struct.unpack_from("<iiii", body, pos + STC_NAME_BYTES)
        entries.append(StcEntry(name, start, end, num, span))
    return entries


# ----------------------------------------------------------------------------
# .snc
# ----------------------------------------------------------------------------
class SyncTable:
    """Samplestamp -> POSIX seconds, as ``snc_reader.find_snc_for_samplestamp``."""

    def __init__(self, stamps: np.ndarray, unix_usec: np.ndarray):
        self.stamps = stamps  # int64
        self.unix_usec = unix_usec  # int64

    def __len__(self):
        return self.stamps.size

    def stamp_to_posix(self, target_stamp: int, sample_freq: float) -> float | None:
        if self.stamps.size == 0:
            return None
        # nearest entry at or before the target, scanning in file order and
        # stopping at the first entry past it (as the reference does)
        past = self.stamps > target_stamp
        k = int(np.argmax(past)) if past.any() else self.stamps.size
        if k > 0:
            best = k - 1
            return int(self.unix_usec[best]) / 1e6 + (target_stamp - int(self.stamps[best])) / sample_freq
        return int(self.unix_usec[0]) / 1e6 - (int(self.stamps[0]) - target_stamp) / sample_freq


def read_snc_file(snc_path) -> SyncTable:
    with open(snc_path, "rb") as f:
        raw = f.read()
    data = raw[GENERIC_HEADER_END:]
    n = len(data) // SNC_ENTRY_SIZE
    dt = np.dtype({"names": ["stamp", "ft"], "formats": ["<u4", "<u8"], "offsets": [0, 4], "itemsize": 12})
    arr = np.frombuffer(data, dtype=dt, count=n)
    stamps = arr["stamp"].astype(np.int64)
    usec = (arr["ft"] // 10).astype(np.int64) - FILETIME_UNIX_DIFF * 1_000_000
    return SyncTable(stamps, usec)


# ----------------------------------------------------------------------------
# .ent: channel names and NtKeys records
# ----------------------------------------------------------------------------
def parse_montage_channel_names(ent_bytes: bytes) -> list[str] | None:
    """Channel names of the last ``ChanNames`` list in the .ent (as xltek2mef)."""
    loc = ent_bytes.rfind(b"ChanNames")
    if loc < 0:
        return None
    open_paren = ent_bytes.find(b"(", loc)
    if open_paren < 0:
        return None
    end = ent_bytes.find(b"))", open_paren)
    if end < 0:
        return None
    text = ent_bytes[open_paren : end + 2].decode("utf-8", errors="ignore")
    names = []
    for name in re.findall(r'"([^"]*)"', text):
        name = name.replace(" ", "-").replace("/", "-")
        if name:
            names.append(name)
    return names


class _Node:
    __slots__ = ("key", "data")

    def __init__(self, key, data):
        self.key = key
        self.data = data

    def to_dict(self):
        if isinstance(self.data, list):
            return {item.key: item.to_dict() for item in self.data}
        return self.data


def _parse_key_tree(t, start=0, end=-1):
    """Parse an NtKeys parenthesized key-value tree (port of exportnatusevents)."""
    nodes = []
    found_key = False
    cursor = start
    stopping = len(t) if end == -1 else end
    while cursor < stopping:
        if cursor + 3 < len(t) and t[cursor : cursor + 3] == '(."' and t[cursor + 3] not in [")", "("]:
            found_key = True
            cursor += 3
        if not found_key:
            cursor += 1
            continue
        next_quote = t.find('"', cursor)
        if next_quote == -1:
            break
        key = t[cursor:next_quote]
        if next_quote + 3 >= len(t):
            break
        if t[next_quote + 3] != "(":
            next_paren_right = t.find(")", cursor)
            if next_paren_right == -1:
                break
            while next_paren_right + 1 < len(t) and t[next_paren_right + 1] not in [",", ")"]:
                next_paren_right = t.find(")", next_paren_right + 1)
                if next_paren_right == -1:
                    break
            if next_paren_right == -1:
                break
            val = t[next_quote + 3 : next_paren_right].strip('"')
            if val.isdigit():
                val = int(val)
            else:
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    pass
            nodes.append(_Node(key, val))
        else:
            inside_text = False
            count = None
            cursor_inner = next_quote + 3
            while count != 0:
                if count is None:
                    count = 0
                if cursor_inner >= len(t):
                    break
                if t[cursor_inner] == '"':
                    inside_text = not inside_text
                if t[cursor_inner] == "(" and not inside_text:
                    count += 1
                elif t[cursor_inner] == ")" and not inside_text:
                    count -= 1
                cursor_inner += 1
            next_paren_right = cursor_inner
            nodes.append(_Node(key, _parse_key_tree(t, start=next_quote + 3, end=cursor_inner + 1)))
        cursor = next_paren_right + 1
        found_key = False
    return nodes


def parse_ntkeys(text: str) -> dict:
    result = {node.key: node.to_dict() for node in _parse_key_tree(text)}
    if "Stamp" in result and isinstance(result["Stamp"], float):
        result["Stamp"] = int(result["Stamp"])
    return result


def parse_ent_records(ent_bytes: bytes, data_types: set[str]) -> list[dict]:
    """Records of the .ent whose ``Data.Type`` is in ``data_types``."""
    markers = [t.encode("utf-8") for t in data_types]
    records = []
    pos = GENERIC_HEADER_END
    n = len(ent_bytes)
    while pos + 16 <= n:
        note_type, rec_len, _prev_len, _const = struct.unpack_from("<iiii", ent_bytes, pos)
        if note_type == 0 or rec_len < 16:
            break
        end = pos + rec_len
        if end > n:
            break
        payload = ent_bytes[pos + 16 : end]
        null_idx = payload.find(b"\x00")
        if null_idx >= 0:
            payload = payload[:null_idx]
        if any(m in payload for m in markers):
            try:
                record = parse_ntkeys(payload.decode("utf-8", errors="replace"))
            except Exception:
                record = None
            if record is not None:
                data = record.get("Data")
                if isinstance(data, dict) and data.get("Type") in data_types:
                    records.append(record)
        pos = end
    return records


def parse_filetime_hex(hex_str) -> int | None:
    """FILETIME hex string ('0x00000008' + 16 hex digits, little endian) -> int."""
    if not isinstance(hex_str, str) or not hex_str.startswith("0x") or len(hex_str) < 26:
        return None
    try:
        return struct.unpack("<Q", bytes.fromhex(hex_str[10:]))[0]
    except (ValueError, struct.error):
        return None


# ----------------------------------------------------------------------------
# folder listing
# ----------------------------------------------------------------------------
@dataclass
class SessionFiles:
    erd: list[Path]  # sorted by name
    etc: dict[str, Path]  # by lower-case stem
    stc: Path | None
    snc: Path | None
    ent: Path | None
    sizes: dict[str, int]  # file name -> size, for every listed file


def list_session_files(dirname) -> SessionFiles:
    """List the session folder once (subfolders such as Decimated are ignored)."""
    dirname = Path(dirname)
    erd, etc, stc, snc, ent = [], {}, [], [], []
    sizes = {}
    with os.scandir(dirname) as it:
        for entry in it:
            if not entry.is_file():
                continue
            stem, suffix = os.path.splitext(entry.name)
            suffix = suffix.lower()
            if suffix not in (".erd", ".etc", ".stc", ".snc", ".ent"):
                continue
            path = dirname / entry.name
            sizes[entry.name] = entry.stat().st_size
            if suffix == ".erd":
                erd.append(path)
            elif suffix == ".etc":
                etc[stem.lower()] = path
            elif suffix == ".stc":
                stc.append(path)
            elif suffix == ".snc":
                snc.append(path)
            elif suffix == ".ent":
                ent.append(path)
    return SessionFiles(
        erd=sorted(erd),
        etc=etc,
        stc=sorted(stc)[0] if stc else None,
        snc=sorted(snc)[0] if snc else None,
        ent=sorted(ent)[0] if ent else None,
        sizes=sizes,
    )

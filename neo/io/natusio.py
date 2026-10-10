"""
IO for reading Natus/XLTEK (NeuroWorks) EEG session folders.

See :class:`neo.rawio.NatusRawIO`. Needs numba (``pip install neo[natus]``).
"""

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.natusrawio import NatusRawIO


class NatusIO(NatusRawIO, BaseFromRaw):
    """
    IO for reading a Natus/XLTEK (NeuroWorks) EEG session folder.
    """

    name = "Natus IO"
    description = "IO for reading Natus/XLTEK NeuroWorks session folders"

    _prefered_signal_group_mode = "group-by-same-units"
    mode = "dir"

    def __init__(self, dirname=""):
        NatusRawIO.__init__(self, dirname=dirname)
        BaseFromRaw.__init__(self, dirname)

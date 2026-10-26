"""Opt-in strict decoding must not fabricate an incomplete native sample."""
import numpy as np
import pytest
from neo.rawio.natusrawio.erd_decode import decode_range, ErdLayout

@pytest.mark.parametrize("payload", [[0,0,1], [0,3,255,255,255,255,1,0,0,0,2,0]])
def test_strict_rejects_incomplete_packet(payload):
    data=np.asarray(payload,dtype=np.uint8)
    layout=ErdLayout(2,[1,2])
    assert decode_range(data,layout,0,1).shape==(1,2)
    assert decode_range(data,layout,0,1,strict=True).shape==(0,2)

def test_strict_keeps_complete_packet():
    data=np.asarray([0,0,1,2],dtype=np.uint8)
    np.testing.assert_array_equal(decode_range(data,ErdLayout(2,[1,2]),0,1,strict=True),[[1,2]])

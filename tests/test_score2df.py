
from score2df import *

def test_scoquire2df():
    file = "midi/sonate_31.mid"
    df = score2df(file)
    assert df.shape[0] > 0

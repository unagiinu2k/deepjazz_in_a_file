from os import *
from score2df import *

def test_scoquire2df():
    file = "midi/sonate_31.mid"
    df = score2dataframe(file)
    assert df.shape[0] > 0

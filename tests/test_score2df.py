#from os import *
from score2df import *
#import numpy as np

#file = "midi/sonate_31.mid"
#file = 'chorales/midi/065900b_.mid'

def test_score2df():
    file = "midi/sonate_31.mid"
    df = score2dataframe(file)
    assert df.shape[0] > 0



def test_sequencial_coding():
    file = "midi/sonate_31.mid"
    df = score2dataframe(file)
    
    df = add_sequential_diffs(df)
    if False:
            df.sort_values(['time' , 'n'] , inplace = True)
            df = df.assign(dt = df.time.diff())
            df = df.assign(dcent = np.where(df.n == 0 , df.groupby('n').cent.diff(), df.groupby('time').cent.diff()))    
    

    df.head()
    assert df.shape[0] > 0

def test_yx_encoder():
    a_list = [['a','c','d'] , ['a' , 'd', 'e' , 'b']]

    y,x = yx_encoder(a_list)
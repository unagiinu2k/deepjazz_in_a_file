#from os import *
import score2df 
#import numpy as np

#file = "midi/sonate_31.mid"
#file = 'chorales/midi/065900b_.mid'

def test_score2df():
    file = "midi/sonate_31.mid"
    df = score2df.score2dataframe(file)
    assert df.shape[0] > 0

def test_flat_structured_midi():
    file = 'chorales/midi/028100b_.mid'
    df = score2df.score2dataframe(file)
    assert df.shape[0] > 0


def test_sequencial_coding():
    file = "midi/sonate_31.mid"
    df = score2df.score2dataframe(file)
    
    df = score2df.add_sequential_diffs(df)
    if False:
            df.sort_values(['time' , 'n'] , inplace = True)
            df = df.assign(dt = df.time.diff())
            df = df.assign(dcent = np.where(df.n == 0 , df.groupby('n').cent.diff(), df.groupby('time').cent.diff()))    
    

    df.head()
    assert df.shape[0] > 0
    batch_X = ppd_X[0][: , batch_samples]
    batch_y = ppd_y[0][:,  batch_samples]        
    batch_mask = mask[batch_samples]
    
    #if is_use_gpu:
    #    batch_X = try_gpu(batch_X)
    #    batch_y = try_gpu(batch_y)
    batch_X = batch_X.to(device)
    batch_y = batch_y.to(device)

    batch_msk = batch_mask.to(device)# = try_gpu(batch_mask)

    model.zero_grad()

    batch_y_model = model(batch_X)
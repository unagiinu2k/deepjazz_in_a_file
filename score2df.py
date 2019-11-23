from music21 import converter, instrument, note, chord
from music21 import interval
import pandas as pd
import numpy as np
from itertools import chain
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch

def score2dataframe(file):
    """
    create data frame from midi file
    
    Parameters
    ----------
    file : string
        file path to a midi file
    
    Returns
    -------
    pandas data frame whose columns are
        pitch : pitch of each note
        time : time from the start of each note
        cent : pitch as interger relative to C4
        n : for example, 1 if a note is the second from the lowest among the simultaneously pressed notes
        dcent: the difference of cent from the previous note after grouping by n

    """
    
    midi = converter.parse(file)
    notes_to_parse = None
    max_simultaneous = 4
    #pitches = [[] for i in range(max_simultaneous)]
    #times = [[] for i in range(max_simultaneous)]
    #diffs = [[] for i in range(max_simultaneous)]
    pitches = []
    times = []
    diffs = []

    parts = instrument.partitionByInstrument(midi)
    if parts: # file has instrument parts
        notes_to_parse = parts.parts[0].recurse()
    else: # file has notes in a flat structure
        notes_to_parse = midi.flat.notes
    for element in notes_to_parse:
        
        if isinstance(element, note.Note):        
            #run_notes = [element]
            pitches.append(str(element.pitch))
            diffs.append(interval.notesToChromatic(note.Note("C4") , element).cents)
            times.append(float(element.offset))
        elif isinstance(element, chord.Chord):           
            run_chord = element
            for i in range(min(len(element.normalOrder) , max_simultaneous)):
                pitches.append(element.pitches[i])
                diffs.append(interval.notesToChromatic(note.Note("C4") , run_chord.notes[i]).cents)
                times.append(float(element.offset))


    df_score = pd.DataFrame({'pitch' : pitches , 'time' : times , 'cent' : diffs })
    df_score.sort_values(['time' , 'cent'] , inplace=True)

    df_score = df_score.assign(n = df_score.groupby('time').cumcount())

    df_score = df_score.assign(dcent = df_score.groupby('n').cent.diff())
    return df_score


def add_sequential_diffs(df):
    """
    add Columns dcent and dt consistent with the 
    sequential chords-notes coding idea
    Parameters
    ----------
    df : data frame
        
    """
    df.sort_values(['time' , 'n'] , inplace = True)
    df = df.assign(dt = df.time.diff())
    df = df.assign(dcent = np.where(df.n == 0 , df.groupby('n').cent.diff(), df.groupby('time').cent.diff()))
    return df



def add_lags(df):
    """[summary]
    
    Parameters
    ----------
    df : pandas.dataframe corresponding to a score
        
    """
    
    df = df.assign(dcent = df.groupby('n').cent.diff())
    df = df.assign(dcent_lag1 = df.groupby('n').dcent.shift(1))
    df = df.assign(dcent_lag2 = df.groupby('n').dcent.shift(2))
    df = df.assign(dcent_lag3 = df.groupby('n').dcent.shift(3))

    df = df.assign(dtime = df.groupby('n').time.diff())
    #df = df.assign(dcent_lag1 = df.dcent.shift(1))
    #df = df.assign(dcent_lag2 = df.dcent.shift(2))
    #df = df.assign(dcent_lag3 = df.dcent.shift(3))
    return df


def yx_encoder(notes_list):
    """ returns torch.tensor's y and x appropriate for constructing prediction model from list of list of characters 

    
    Parameters
    ----------
    notes_list : list
        list of list which stores time series of characters
    
    Returns
    -------
    [type]
        [description]
    """
    le = LabelEncoder()
    note_set = set(chain.from_iterable(notes_list))

    le.fit(list(note_set))

    labeled_notes_list = [le.transform(np.array(x)) for x in notes_list]

    label_set = set(le.transform(list(note_set)))

    raw_X = [torch.zeros(labeled_notes_list[i].shape[0] - 1 , len(label_set)) for i in range(len(notes_list))]

    for i in range(len(notes_list)):
        for j in range(labeled_notes_list[i].shape[0]-1):
            raw_X[i][j , labeled_notes_list[i][j]] = 1.

    raw_y = [torch.tensor(np.array(x[1:])) for x in labeled_notes_list]

    return raw_y , raw_X
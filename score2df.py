from music21 import converter, instrument, note, chord
from music21 import interval
import pandas as pd
#file = "midi/sonate_31.mid"
#file = 'chorales/midi/065900b_.mid'

def score2df(file):
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
            times.append(element.offset)
        elif isinstance(element, chord.Chord):           
            run_chord = element
            for i in range(min(len(element.normalOrder) , max_simultaneous)):
                pitches.append(element.pitches[i])
                diffs.append(interval.notesToChromatic(note.Note("C4") , run_chord.notes[i]).cents)
                times.append(element.offset)


    df_score = pd.DataFrame({'pitch' : pitches , 'time' : times , 'cent' : diffs })
    df_score.sort_values(['time' , 'cent'] , inplace=True)

    df_score = df_score.assign(n = df_score.groupby('time').cumcount())

    df_score = df_score.assign(dcent = df_score.groupby('n').cent.diff())
    return df_score
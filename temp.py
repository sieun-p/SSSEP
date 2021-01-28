# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import mne
import os,fnmatch

dPath="C:/Users/user/Downloads/SSSEP/SSSEP"
files=fnmatch.filter(os.listdir(dPath),'*.set')
raw = mne.io.read_raw_eeglab(files[0])
raw.plot()
events, event_id = mne.events_from_annotations(raw)

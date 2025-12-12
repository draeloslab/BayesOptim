from simulate.neuron import Neuron

import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import random
import ast
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

class PseudoNeurons(Neuron):
    ''' 
    Class to generate neural tuning curves and responses to visual stimuli from experimental data
    '''   
    #TODO: PseudoNeuron shoudl augment experimental data using realsitic noise conditions
    def __init__(self, c_array, stim_df, stim_conditions, peaks, pre_stim_window, stim_extension, tol=1):
        ''' 
        Initialize the PseudoNeurons instance

        params:
        ----------
        c_array                  : neural trace against time
        stim_df (df)     : dataframe containing the visual stimuli conditions and the corresponding stimulus frames     
        onset_offset_data (dict) : stimulus onset and offset pairs
        peaks                    : true peaks (generated from offline fit)
        pre_stim_window (int)    : window before stimulus
        stim_extension (int)     : window after stimulus
        tol (tuple)              : distance tolerance for peak locations
        '''
        super().__init__(len(c_array), peaks.shape[1], tol)
        self.c_array = c_array
        self.stim_df = stim_df  # stimulus_df (dataframe)
        self.on_times = stim_conditions['Onset'].to_list()
        self.off_times = stim_conditions['Offset'].to_list()
        self.pre_stim_window = pre_stim_window
        self.stim_extension = stim_extension

        self.peaks = peaks
        self.y = []

    def choose_stim_idx(self, stim_sample):
        ''' 
        Choose a random stimulus to sample
        
        params:
        ----------
        stim_sample (tuple) : stimulus sample
        '''
        filtered_df = self.stim_df[(self.stim_df['Funkiness'] == stim_sample[0]) &
                                    (self.stim_df['Orientation (V/H)'] == stim_sample[1]) &
                                    (self.stim_df['Contrast'] == stim_sample[2])]
        if type(filtered_df['Pairs'].iloc[0]) == str:
            filtered_df['Pairs'] = filtered_df['Pairs'].apply(ast.literal_eval) ## a value is trying to be set on a copy of a slice from a DF - try using .iloc[row, col] = value
        pairs_list = filtered_df['Pairs'].iloc[0] 
        x_sample = random.choice(pairs_list)    # stim_idx

        return x_sample

    def sample(self, stim_sample):
        ''' 
        Samples a z response an x sample stimulus
        
        params:
        ----------
        x (tuple)        : sample stimulus
        normalize (bool) : if True, normalize response by the maximum response

        returns:
        ----------
        z                : area under the neural trace curve = response given a sample stimulus
        '''
        x_sample = self.choose_stim_idx(stim_sample)
        stim_idx = self.on_times.index(x_sample[0])

        frame = [self.on_times[stim_idx], self.off_times[stim_idx]]
        responses = self.c_array[:, frame[0]-self.pre_stim_window:frame[1]+self.stim_extension]
        z = np.sum(responses, axis=1)   # auc for all neurons for this stim_idx

        self.record_response(x_sample, z)

        return z


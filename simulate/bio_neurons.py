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

class BioNeurons(Neuron):
    ''' 
    Class to generate neural tuning curves and responses to visual stimuli from experimental data
    '''   
    #TODO: PseudoNeuron should augment experimental data using realsitic noise conditions
    def __init__(self, c_array, stim_df, stim_conditions, peaks, pre_stim_window, stim_extension, offline_f, stimY, good_neuron_idx_ls=None, tol=1, seed=42, lock_trial=False):
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
        offline_f                : array of predicted posterior mean for all neurons (the offline fit)
        tol (tuple)              : distance tolerance for peak locations
        '''
        super().__init__(len(c_array), np.array(peaks).shape[1], tol)
        self.c_array = c_array
        self.stim_df = stim_df  # stimulus_df (dataframe)
        self.on_times = stim_conditions['Onset'].to_list()
        self.off_times = stim_conditions['Offset'].to_list()
        self.pre_stim_window = pre_stim_window
        self.stim_extension = stim_extension  # believe the unit is frame
        self.offline_f = offline_f
        self.stimY = stimY
        self.good_neuron_idx_ls = good_neuron_idx_ls

        self.peaks = peaks
        
        self.resp_x = []
        self.resp_z = []
        self.y = []
        print(f"hey new PseudoNeurons here with seed {seed}!")

        # === MINIMAL CHANGE ===
        self._base_seed = int(seed) if seed is not None else None
        self.rng = np.random.default_rng(self._base_seed)  # one run-level RNG
        self.lock_trial = lock_trial
        self._trial_cache = {}          # lock chosen trial per stimulus
        self._stim_repeat_counter = {}  # count repeats per stimulus for noise

    def record_response(self, x, z):
        ''' 
        Records z response given an x sample

        params:
        ----------
        x (tuple)        : sample stimulus
        z                : simulated response given sample stimulus
        normalize (bool) : if True, normalize response by the maximum response

        '''
        self.resp_x.append(x)
        self.resp_z.append(z)
    def choose_stim_idx(self, stim_sample):
        ''' 
        Choose a random stimulus to sample
        Return on off stimulus sets for AUC integration if the stimulus was used; otherwise returned the origincal stim_sample 
        
        params:
        ----------
        stim_sample (tuple) : stimulus sample
        '''

        if len(stim_sample) == 3:
            filtered_df = self.stim_df[(self.stim_df['Funkiness'] == stim_sample[0]) &
                                        (self.stim_df['Orientation (V/H)'] == stim_sample[1]) &
                                        (self.stim_df['Contrast'] == stim_sample[2])]
        elif len(stim_sample) == 4:
            filtered_df = self.stim_df[(self.stim_df['Angle'] == stim_sample[0]) &
                                        (self.stim_df['Speed'] == stim_sample[1]) &
                                        (self.stim_df['Size'] == stim_sample[2]) &
                                        (self.stim_df['Frequency'] == stim_sample[3])]
        elif len(stim_sample) == 5:
            filtered_df = self.stim_df[(self.stim_df['Angle'] == stim_sample[0]) &
                                        (self.stim_df['Speed'] == stim_sample[1]) &
                                        (self.stim_df['Size'] == stim_sample[2]) &
                                        (self.stim_df['Frequency'] == stim_sample[3]) &
                                        (self.stim_df['Contrast'] == stim_sample[4])]
        if type(filtered_df['Pairs'].iloc[0]) == str:  # doesn't seemed to be runned
            filtered_df['Pairs'] = filtered_df['Pairs'].apply(ast.literal_eval) ## a value is trying to be set on a copy of a slice from a DF - try using .iloc[row, col] = value
        pairs_list = filtered_df['Pairs'].iloc[0]
        if not pairs_list:  # if it's empty
            self.empty = True
            x_sample = None
            # return stim_sample
            return stim_sample, x_sample
        self.empty = False
        stim_key = tuple(stim_sample)
        if self.lock_trial:
            # fixed trial per stimulus (old behavior)
            if stim_key not in self._trial_cache:
                choice_idx = self.rng.integers(0, len(pairs_list))
                self._trial_cache[stim_key] = pairs_list[choice_idx]
            x_sample = self._trial_cache[stim_key]
        else:
            # NEW: vary trial each call, reproducible across runs via self.rng
            choice_idx = self.rng.integers(0, len(pairs_list))
            x_sample = pairs_list[choice_idx]

        return stim_sample, x_sample

    # === MINIMAL ADDITION: per-stimulus noise that varies with repeat index ===
    def _noise(self, size, stim_sample):
        stim_key = tuple(stim_sample)
        k = self._stim_repeat_counter.get(stim_key, 0)
        self._stim_repeat_counter[stim_key] = k + 1

        # derive a child RNG from (base_seed, stim_key, repeat); reproducible across runs with same seed
        if self._base_seed is None:
            # no global seed => different each run (still varies per repeat)
            child_rng = np.random.default_rng()
        else:
            ss = np.random.SeedSequence([
                self._base_seed,
                (hash(stim_key) & 0xFFFFFFFF),
                k
            ])
            child_rng = np.random.default_rng(ss)

        return np.abs(child_rng.normal(0, 0.05, size=size)) + 1e-14
    
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
        stim_sample, x_sample = self.choose_stim_idx(stim_sample) # x_sample = self.choose_stim_idx(stim_sample)  # stim_sample is the stimulus combo?
        if self.empty:  # if it's empty
            # x_sample = stim_sample
            # offline_resp = self.offline_f[:, int(x_sample[0]), int(x_sample[1]), int(x_sample[2]),int(x_sample[3]), int(x_sample[4])]
            if len(stim_sample) == 4:
                offline_resp = self.offline_f[:, int(stim_sample[0]), int(stim_sample[1]), int(stim_sample[2]),int(stim_sample[3])]
            elif len(stim_sample) == 5:
                offline_resp = self.offline_f[:, int(stim_sample[0]), int(stim_sample[1]), int(stim_sample[2]),int(stim_sample[3]), int(stim_sample[4])]
            # z = offline_resp   #071725: use offline without noise #+ np.abs(local_rng.normal(0, 0.05, size = self.offline_f.shape[0])) + 1e-14 #offline_resp #np.maximum(offline_resp, offline_resp + np.random.normal(0, 0.05, size = self.offline_f.shape[0]))
            # z = offline_resp + np.abs(local_rng.normal(0, 0.05, size = self.offline_f.shape[0])) + 1e-14 # 080725: test with noised sample
            z = offline_resp + self._noise(size=self.offline_f.shape[0], stim_sample=stim_sample)# 103025: test with new random state setting
        else:  # 071725: add stimY as a guide for neuron response calculation
            print(f"this is stim_sample {stim_sample} this is x_sample: {x_sample}")
            stim_idx = self.on_times.index(x_sample[0])
            frame = [self.on_times[stim_idx], self.off_times[stim_idx]]
            responses = self.c_array[:, frame[0]-self.pre_stim_window:frame[0]+self.stim_extension]
            z = np.mean(responses, axis=1)  # shape: (n_neurons,)
            if self.good_neuron_idx_ls is not None:
                z = z[self.good_neuron_idx_ls]
            # print("this is z.shape:", z.shape)
            # === PATCH START ===
            # Determine how many neurons existed at this stimulus
            stimY_idx = stim_idx - 1  # stimY is offset by 1 (starts at stimulus #2)
            if stimY_idx >= 0:
                # n_detected = len(self.stimY[stimY_idx])
                raw_n_detected = len(self.stimY[stimY_idx])
                if self.good_neuron_idx_ls is not None:
                    # Count how many of the raw detected neurons are actually in our good list
                    n_detected = np.sum(self.good_neuron_idx_ls < raw_n_detected)
                else:
                    n_detected = raw_n_detected
            else:
                n_detected = 0  # Stimulus #1 had no neurons tracked
            # print(f"n_detected = {n_detected}")

            # Replace padded zeros or nans (undetected neurons) with offline fit
            if n_detected < len(z):
                if len(stim_sample) == 4:
                    offline_patch  = self.offline_f[n_detected:, int(stim_sample[0]), int(stim_sample[1]), int(stim_sample[2]), int(stim_sample[3])] 
                elif len(stim_sample) == 5:
                    offline_patch = self.offline_f[n_detected:, int(stim_sample[0]), int(stim_sample[1]), int(stim_sample[2]), int(stim_sample[3]), int(stim_sample[4])]
                z[n_detected:] = offline_patch+ self._noise(size=int(self.offline_f.shape[0]-n_detected), stim_sample=stim_sample) + 1e-14 


        self.record_response(stim_sample, z)
        return z

    def verify_sln(self, peaks, n):
        ''' 
        Compares the predicted peak location to the true peak location 
        
        params:
        ----------
        peaks (list)  : predicted peak location (N,d)
        n (int)       : n-th neuron 

        returns:
        ----------
        dists (tuple) : predicted peaks - true peak of the n-th neuron
        count (int)   : a count of the number of correct predictions for each dimension
        mse (tuple)   : MSE between predicted peak and true peak of the n-th neuron
        '''
        # peaks shape: (N, d)
        dists = np.abs(peaks - self.peaks[n])           
        mse = mean_squared_error(peaks, self.peaks[n])
        count = np.count_nonzero(dists < self.tol)      # count dist within tolerance

        return dists, count, mse


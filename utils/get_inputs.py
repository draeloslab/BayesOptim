import numpy as np
import pandas as pd
import os

from utils.make_dataframes import find_stim_condition, get_stim_conditions_df

class ExperimentalInputs():
    '''
    ExperimentalInputs class preprocesses data from the Savier and Burgess labs
    '''

    def __init__(self, data_path, dataset, pre_stim_window, stim_extension, lab, N, stimY=None, outliers=None):

        self.data_path = data_path
        self.dataset = dataset
        self.pre_stim_window = pre_stim_window
        self.stim_extension = stim_extension
        self.lab = lab
        self.N = N  # number of neurons
        self.stimY = stimY
        self.outliers = outliers
        # if 'burgess' in self.data_path:
        #     self.lab = 'burgess'
        # elif 'savier' in self.data_path:
        #     self.lab = 'savier'  
        # print(self.lab)
        # with open(os.path.join(self.data_path, 'response_data', f'cnm_C_{self.dataset}.npy')) as f:
        #     self.C = np.load(f)  
        if dataset == "":
            self.C = np.loadtxt(os.path.join(self.data_path, 'raw_C.txt'))
            if self.outliers is not None:
                self.good_neuron_idx_ls = np.array(list(set(np.arange(min(self.N + len(self.outliers), self.C.shape[0]))) - set(self.outliers)))
            else:
                self.good_neuron_idx_ls = np.arange(self.N)
            # we do not trim C here
            # # if self.N < self.C.shape[0]:
            # if len(self.good_neuron_idx_ls) < self.C.shape[0]:
            #     self.C = self.C[self.good_neuron_idx_ls]
            #     print(f"Got rid of {len(self.outliers)} outliers. Trimming down self.C to {len(self.good_neuron_idx_ls)} neurons. Current shape of C is {self.C.shape}")
        else:
            self.C = np.load(os.path.join(self.data_path, 'response_data', f'cnm_C_{self.dataset}.npy'))
        self.get_conditions()
        self.get_auc()
        if self.outliers is not None and len(self.good_neuron_idx_ls) < self.C.shape[0]:
            self.C = self.C[self.good_neuron_idx_ls]
            print(f"Got rid of {len(self.outliers)} outliers. Trimming down self.C to {len(self.good_neuron_idx_ls)} neurons. Current shape of C is {self.C.shape}")

    def get_conditions(self):
        '''
        This method loads in the lab's full stimulus conditions CSV file that contains information about the different stimulus types. It then reads in a CSV file that contains information about the 
        stimulus onset and offset periods and their corresponding stimulus condition. This is then merged together with the full condition dataframe to create a dataframe that contains both onset/offset
        frames and the specific corresponding stimulus. 
        '''
        if self.lab == 'savier':
            if self.dataset == 'dg' or self.dataset == 'l' or self.dataset == 'md':
                condition_df = pd.read_csv(os.path.join(self.data_path, f'{self.lab}_fullConditions_{self.dataset}.csv'), usecols=[1,2])
            elif self.dataset == 'fs' or self.dataset == 'ms':
                condition_df = pd.read_csv(os.path.join(self.data_path, f'{self.lab}_fullConditions_{self.dataset}.csv'), usecols=[1,2,3])
            else:
                condition_df = pd.read_csv(os.path.join(self.data_path, f'{self.lab}_fullConditions.csv'))
        elif self.lab == 'burgess':
            if self.dataset == "":
                condition_df = pd.read_csv(os.path.join(self.data_path, f'{self.lab}_fullConditions.csv'))
        
        if self.dataset == "":
            on_off_times = pd.read_csv(os.path.join(self.data_path,'stim_data', f'on_off_times.csv'))
        else:
            on_off_times = pd.read_csv(os.path.join(self.data_path, 'stim_data', f'on_off_times_{self.dataset}.csv'), usecols=[1,2,3])
        
        stim_conditions = find_stim_condition(on_off_times, condition_df)
        # print(stim_conditions)
        self.stim_conditions = stim_conditions[stim_conditions['Onset'] + self.stim_extension < self.C.shape[1]]
        # print(f"self.stim_conditions are {self.stim_conditions}")
        self.on_times = self.stim_conditions['Onset'].to_list()
        self.off_times = self.stim_conditions['Offset'].to_list()

        self.x_stim = self.get_x_stim(self.stim_conditions)
        if self.dataset == "":
            if os.path.isfile(os.path.join(self.data_path, 'stim_data', f'x_stim.npy')):
                pass
            else:
                print("saving a new copy of x_stim...")
                with open(os.path.join(self.data_path, 'stim_data', f'x_stim.npy'), 'wb') as f:
                    np.save(f, self.x_stim)
        else:
            if os.path.isfile(os.path.join(self.data_path, 'stim_data', f'x_stim_{self.dataset}.npy')):
                pass
            else:
                with open(os.path.join(self.data_path, 'stim_data', f'x_stim_{self.dataset}.npy'), 'wb') as f:
                    np.save(f, self.x_stim)

        if self.lab == 'burgess':
            type = self.dataset #'funkystim'
        else:
            type = self.dataset
        # print(self.stim_conditions)
        self.stimulus_df = get_stim_conditions_df(self.stim_conditions.to_dict(), condition_df, type)
        # print(self.stimulus_df)

    def get_x_stim(self, stim_conditions):
        '''
        Generates a numpy array for every stimulus shown.
        '''

        keys = stim_conditions.columns.tolist()
        first_param = keys[2]
        second_param = keys[3] if len(keys) > 3 else None
        third_param = keys[4] if len(keys) > 4 else None
        fourth_param = keys[5] if len(keys) > 5 else None
        fifth_param = keys[6] if len(keys) > 6 else None
        if fifth_param and fourth_param and second_param and third_param:
            x_stim = stim_conditions[[first_param, second_param, third_param, fourth_param, fifth_param]].dropna().values
        elif fourth_param and second_param and third_param:
            x_stim = stim_conditions[[first_param, second_param, third_param, fourth_param]].dropna().values
        elif second_param and third_param:
            x_stim = stim_conditions[[first_param, second_param, third_param]].dropna().values
        elif second_param:
            x_stim = stim_conditions[[first_param, second_param]].dropna().values
        else:
            x_stim = stim_conditions[[first_param]].dropna().values
        
        return x_stim

    def get_auc(self):  #TODO: change this to how we calculate neuron response in improv
        '''
        Calculates the area under the curve (AUC) for each neuron during every stimulus window. 
        '''
        self.auc = np.empty((len(self.C), len(self.on_times)))
        # print(self.C[0, :100])
        print("pre_stim_window is", self.pre_stim_window, "; stim_extension is", self.stim_extension)
        # with open(os.path.join(self.data_path,'response_data', f'summed_response_without_proc.npy'), 'wb') as f:
        #     np.save(f, self.auc)
        # print(self.on_times)
        for neuron in range(len(self.C)):
            for i in range(len(self.on_times)):
                if self.on_times[i]-self.pre_stim_window < 0:
                    frame = [self.on_times[i], self.off_times[i]+self.stim_extension]
                else:
                    # frame = [self.on_times[i]-self.pre_stim_window, self.off_times[i]+self.stim_extension]  # 071725: changed it to how we do in improv
                    frame = [self.on_times[i]-self.pre_stim_window, self.on_times[i]+self.stim_extension]
                # summed_responses = np.sum(self.C[neuron, frame[0]:frame[1]], axis=0)
                summed_responses = np.mean(self.C[neuron, frame[0]:frame[1]], axis = 0)
                self.auc[neuron, i] = summed_responses
        if self.stimY is not None:
            print("matching with stimY")
            auc_out = self.auc.astype(float).copy()

            # update 082625: again we dont need to iterate from 1 to n_stimY+1
            # # ---------- 1. deal with column 0 ---------------------------------
            # col0_mask = auc_out[:, 0] == 0
            # auc_out[col0_mask, 0] = np.nan

            # ---------- 2. deal with columns 1-41 ------------------------------
            n_stimY   = len(self.stimY)       # 41
            print(f"n_stimY: {n_stimY}")
            print(f"auc_out shape: {auc_out.shape}")
            n_neurons, num_stimuli = auc_out.shape # 498

            for s_idx in range(min(num_stimuli, n_stimY)):  # update 092725 (and earlier): we use mask_invalid not mask_padding algo now
                n_detected = len(self.stimY[s_idx])
                if n_detected < n_neurons:
                    auc_out[n_detected:, s_idx] = np.nan
            # for col in range(n_stimY): # update 082625: again we dont need to iterate from 1 to n_stimY+1
            #     n_detected = len(self.stimY[col])
            #     if n_detected < n_neurons:
            #         rows_to_check = slice(n_detected, None)
            #         zeros = auc_out[rows_to_check, col] == 0
            #         auc_out[rows_to_check, col][zeros] = np.nan
            self.auc = auc_out

        # Trim outliers here at the end before saving so it doesn't mess up stimY mapping
        if len(self.good_neuron_idx_ls) < self.auc.shape[0]:
            print(f"Filtering down to {len(self.good_neuron_idx_ls)} good neurons.")
            self.auc = self.auc[self.good_neuron_idx_ls]

        if os.path.isfile(os.path.join(self.data_path,'response_data', f'summed_response_{self.dataset}.npy')):
            print("summed_response_ file already exists")
            pass
        else:
            if not os.path.exists(os.path.join(self.data_path,'response_data')):
                os.makedirs(os.path.join(self.data_path,'response_data'))
                print("created response_data directory")
            with open(os.path.join(self.data_path,'response_data', f'summed_response_{self.dataset}.npy'), 'wb') as f:
                np.save(f, self.auc)


        

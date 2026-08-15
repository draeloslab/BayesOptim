from scipy import io
import numpy as np
import pandas as pd
import random
import os

def create_fullConditions_df(csv_path, mat_file=None, log_file=None, type=None):
    '''
    Creates dataframe of Burgess Lab funky stim conditions and the Savier's stim conditions for Drift Gratings, Looming, Flashing Spots, Moving Dots, Moving Spots

    input:
    mat_file: (str) Path to the mat file containing full set of conditions
    csv_path: (str) Path to the location you want to save the csv file
    log_file: (str) Path to the location of the log file containing stimulus information for Flashing Spots and Moving Spots (optional)
    type:     (str) Dataset type. This is what identifies the different types of stimuli.

    returns:
    condition_df:  (df) Dataframe containing all 16 funky stim conditions
    '''
    if 'savier' in csv_path:
        if type == 'dg':
            degrees = list(range(0,331,30))
            condition_df = pd.DataFrame({'Index': range(1, len(degrees) + 1), 'Degrees': degrees})
        if type == 'fs':
            f = open(log_file, 'r')
            logs = f.read().split('\n')
            logs_filtered = [x for x in logs if "PositionX:" in x]
            conditions = []
            x_pos = []
            y_pos = []
            for log in logs_filtered:
                conditions.append(int(float(log.split("Index:")[0].split()[3])))
                x_pos.append(int(float(log.split("PositionX:")[1].split()[0])))
                y_pos.append(int(float(log.split("PositionY:")[1].split()[0])))

            fs_conditions_df = pd.DataFrame({'Index': conditions, 'x_pos': x_pos, 'y_pos': y_pos})
            pause = pd.DataFrame({'Index': [50], 'x_pos': [np.NaN], 'y_pos': [np.NaN]})
            fs_conditions_df = pd.concat([fs_conditions_df.loc[:46], pause, fs_conditions_df.loc[47:]]).reset_index(drop=True)
            fs_conditions_df = pd.concat([fs_conditions_df.loc[:69], pause, fs_conditions_df.loc[70:]]).reset_index(drop=True)

            condition_df = fs_conditions_df.sort_values(by=['Index'], ignore_index=True)
            condition_df = condition_df.drop_duplicates().reset_index(drop=True)
        if type == 'l':
            condition_df = pd.DataFrame({'Index': range(1,5), 'Speed': np.arange(10,50,10)})
        if type == 'md':
            condition_df = pd.DataFrame({'Index': range(1,13), 'Direction': np.arange(0, 331, 30)})
        if type == 'ms':
            f = open(log_file, 'r')
            logs = f.read().split('\n')
            logs_filtered = [x for x in logs if "Orientation:" in x]

            conditions = []
            orientation = []
            y_range = []
            for log in logs_filtered:
                conditions.append(int(float(log.split("Condition:")[0].split()[3])))
                orientation.append(int(float(log.split("Orientation:")[1].split()[0])))
                y_range.append(int(float(log.split("Y_Range_1:")[1].split()[0])))

            ms_conditions_df = pd.DataFrame({'Index': conditions, 'Orientation': orientation, 'y_range': y_range})

            condition_df = ms_conditions_df.sort_values(by=['Index'], ignore_index=True)
            condition_df = condition_df.drop_duplicates().reset_index(drop=True)
        condition_df.to_csv(os.path.join(csv_path, f'savier_fullConditions_{type}.csv'))

    else:
        stim_dataset = io.loadmat(mat_file)['s']['StimOnset'][0][0][0][0]

        full_conditions = []
        for i in range(len(stim_dataset[13][0][0])):
            full_condition = stim_dataset[13][0][0][i][5]
            full_conditions.append(full_condition)

        full_stim = np.empty([16, 4])
        for i in range(len(full_conditions)):
            full_stim[i,0] = i + 1
            if "Normal" in full_conditions[i][0]:
                full_stim[i,1] = 0
            elif "Funky" in full_conditions[i][0]:
                full_stim[i,1] = 1
                if "Funky_2" in full_conditions[i][0]:
                    full_stim[i,1] = 2
                elif "Funky_3" in full_conditions[i][0]:
                    full_stim[i,1] = 3
        for i in range(len(full_conditions)): 
            if "Vertical" in full_conditions[i][0]:
                full_stim[i,2] = 0
            elif "Horizontal" in full_conditions[i][0]:
                full_stim[i,2] = 1
        for i in range(len(full_conditions)): 
            if "50C" in full_conditions[i][0]:
                full_stim[i,3] = 1

        condition_df = pd.DataFrame(full_stim, dtype = int, columns = ['Index', 'Funkiness', 'Orientation (V/H)', 'Contrast'])
        condition_df.to_csv(os.path.join(csv_path,'burgess_fullConditions.csv'))

    return condition_df

def create_stim_df(mat_file, csv_path, lab, condition_mat_file=None, create_csv=False):
    '''
    Creates dataframe of stimulation parameters along with onset/offset of stimulations times (frame numbers)

    input:
    mat_file  : (str) Path to the mat file containing full set of conditions; for Savier lab this will be a list of strings
    csv_path  : (str) Path to the location you want to save the csv file
    lab       : (str) Lab dataset, either 'burgess' or 'savier'
    create_csv: (boolean, optional) If true, a CSV file of the dataframe will be saved at the csv_path location. Default is False.

    returns:
    on_off_times: (df) Dataframe containing onset/offset stimulus times and the stimulus condition tag

    '''
    if lab == 'burgess':
        stim_dataset = io.loadmat(mat_file)
        if "nidaq" in mat_file:
            condition_mat_file = io.loadmat(condition_mat_file)['ConditionNumber']
            # Onset/Offset Times
            onset = stim_dataset['trialonsets'][0]
            offset = stim_dataset['trialoffsets'][0]
            condition_by_frame = stim_dataset['visstim'][0]
            conditions = condition_by_frame[onset]
            dataset = mat_file[31:45]

        else:
            # Onset/Offset Times
            dataset = stim_dataset['s']['StimOnset'][0][0][0]
            onset = dataset['trialonsets'][0][0].flatten()
            offset = dataset['trialoffsets'][0][0].flatten()
            conditions = np.squeeze(dataset['condition'][0])
            dataset = mat_file[31:39]
    else:
        mats = []
        for file in mat_file:
            mats.append(io.loadmat(file))
        conditions = mats[0]['ConditionNumber'].reshape(-1).tolist()
        relativeTime = mats[1]['relativeTime'][0]
        on_times = mats[2]['StimulusOnsetsTime']
        off_times = mats[2]['StimulusOffsetsTime']

        stim_length = len(on_times)
        onset = []
        offset = []
        for i in range(stim_length):
            onset.append(np.argmax((relativeTime - on_times[i]) > 0))
            offset.append(np.argmax((relativeTime - off_times[i]) > 0))
        
        stim_types = {
            'DriftGratings': 'dg',
            'FlashingSpot': 'fs',
            'MovingDots': 'md',
            'MovingSpots': 'ms',
            'Looming': 'l',
            'Disk': 'l',
            'Bars': 'ms'
        }
        dataset = next((value for key, value in stim_types.items() if key in mat_file[0]), None)

    on_off_times = pd.DataFrame({'Onset': onset, 'Offset': offset, 'Conditions': conditions})

    if create_csv:
        on_off_times.to_csv(os.path.join(csv_path, 'stim_data', f'on_off_times_{dataset}.csv'))

    return on_off_times

def find_stim_condition(on_off_times, condition_df):
    '''
    Creates an updated dataframe of specific stimulation parameters along with onset/offset of stimulations times (frame numbers)

    input:
    on_off_times: (df) Dataframe containing onset/offset stimulus times and the stimulus condition tag. Can be either dataframe or path to CSV file.
    condition_df: (df) Dataframe containing all 16 funky stim conditions. Can be either dataframe or path to CSV file.
    
    returns:
    stim_conditons: (df) Dataframe containing specific stimulation parameters along with onset/offset of stimulations times (frame numbers)
    '''

    if type(on_off_times) == str:
        on_off_times = pd.read_csv(on_off_times, usecols=[1,2,3])
    if type(condition_df) == str:
        condition_df = pd.read_csv(condition_df, usecols=[1,2,3,4])

    stim_conditions = pd.merge(on_off_times, condition_df, left_on='Conditions', right_on='Index', how='left')
    stim_conditions = stim_conditions.drop(columns=['Conditions', 'Index'])

    return stim_conditions

def combine_stim_pairs(condition_df, onset_offset):
    '''
    This takes the list onset_offset (which is a list of 16 lists. Each list corresponds to the onset/offset frames for a specific stimulus) 
    and combines it with the full list of stimulus types
    '''
    df = pd.DataFrame({'Index': range(1, len(onset_offset) + 1), 'Pairs': onset_offset})
    stim_df = pd.merge(condition_df, df, on='Index', how='left')

    if "Index" in stim_df.columns:
        stim_df = stim_df.drop(columns=['Index'])

    return stim_df

def get_stim_conditions_df(stim_conditions_dict, condition_df=None, type=None):
    '''
    This finds all of the onset/offset pairs corresponding to a specific stimulus and adds them to a list. This will be a list of 16 lists to later index into. 
    This then uses the combine_stim_pairs() function to create a dataframe that contains this information.
    '''

    if type == 'funkystim':
        conditions = [[row['Funkiness'], row['Orientation (V/H)'], row['Contrast']] for index, row in condition_df.iterrows()]
        
        onset_offset = [[] for _ in range(16)]
        for i in range(len(conditions)):
            for key, value in stim_conditions_dict['Funkiness'].items():
                if value == conditions[i][0] and stim_conditions_dict['Orientation (V/H)'][key] == conditions[i][1] and stim_conditions_dict['Contrast'][key] == conditions[i][2]:
                    onset_offset[i].append((stim_conditions_dict['Onset'][key], stim_conditions_dict['Offset'][key]))

        stimulus_df = combine_stim_pairs(condition_df, onset_offset)
    
    if type == "":
        print(f"type is none")
        # print(stim_conditions_dict)
        if "Contrast" in stim_conditions_dict.keys():
            conditions = [[row['Angle'], row['Speed'], row['Size'], row['Frequency'], row['Contrast']] for index, row in condition_df.iterrows()]
            onset_offset = [[] for _ in range(2880)] # TODO: need to be more modular
        else:
            conditions = [[row['Angle'], row['Speed'], row['Size'], row['Frequency']] for index, row in condition_df.iterrows()]
            onset_offset = [[] for _ in range(960)]
        
        for i in range(len(conditions)):
            for key, value in stim_conditions_dict['Angle'].items():
                if "Contrast" in stim_conditions_dict.keys():
                    if value == conditions[i][0] and stim_conditions_dict['Speed'][key] == conditions[i][1] \
                        and stim_conditions_dict['Size'][key] == conditions[i][2] \
                            and stim_conditions_dict['Frequency'][key] == conditions[i][3] \
                                and stim_conditions_dict['Contrast'][key] == conditions[i][4]:
                        onset_offset[i].append((stim_conditions_dict['Onset'][key], stim_conditions_dict['Offset'][key]))
                else:
                    if value == conditions[i][0] and stim_conditions_dict['Speed'][key] == conditions[i][1] \
                        and stim_conditions_dict['Size'][key] == conditions[i][2] \
                            and stim_conditions_dict['Frequency'][key] == conditions[i][3]:
                        onset_offset[i].append((stim_conditions_dict['Onset'][key], stim_conditions_dict['Offset'][key]))

        stimulus_df = combine_stim_pairs(condition_df, onset_offset)

    if type == 'dg':
        stim_pairs = {}
        for i in range(len(stim_conditions_dict["Degrees"])):
            degree = stim_conditions_dict["Degrees"][i]
            if degree not in stim_pairs:
                stim_pairs[degree] = []
            stim_pairs[degree].append((stim_conditions_dict["Onset"][i], stim_conditions_dict["Offset"][i]))

        stimulus_df = pd.DataFrame([(degree, pairs) for degree, pairs in stim_pairs.items()], 
                  columns=["Degrees", "Pairs"])     
    
    if type == 'fs':
        conditions = [[row['x_pos'], row['y_pos']] for index, row in condition_df.iterrows()]
        onset_offset = [[] for _ in range(len(condition_df))]
        for i in range(len(conditions)):
            for key, value in stim_conditions_dict['x_pos'].items():
                if value == conditions[i][0] and stim_conditions_dict['y_pos'][key] == conditions[i][1]:
                    onset_offset[i].append((stim_conditions_dict['Onset'][key], stim_conditions_dict['Offset'][key]))

        stimulus_df = combine_stim_pairs(condition_df, onset_offset) 
    
    if type == 'l':
        stim_pairs = {}
        for i in range(len(stim_conditions_dict["Speed"])):
            degree = stim_conditions_dict["Speed"][i]
            if degree not in stim_pairs:
                stim_pairs[degree] = []
            stim_pairs[degree].append((stim_conditions_dict["Onset"][i], stim_conditions_dict["Offset"][i]))

        stimulus_df = pd.DataFrame([(degree, pairs) for degree, pairs in stim_pairs.items()], 
                  columns=["Speed", "Pairs"])  
    
    if type == 'md':
        stim_pairs = {}
        for i in range(len(stim_conditions_dict["Direction"])):
            degree = stim_conditions_dict["Direction"][i]
            if degree not in stim_pairs:
                stim_pairs[degree] = []
            stim_pairs[degree].append((stim_conditions_dict["Onset"][i], stim_conditions_dict["Offset"][i]))

        stimulus_df = pd.DataFrame([(degree, pairs) for degree, pairs in stim_pairs.items()], 
                  columns=["Direction", "Pairs"])  
    
    if type == 'ms':
        conditions = [[row['Orientation'], row['y_range']] for index, row in condition_df.iterrows()]
        onset_offset = [[] for _ in range(len(condition_df))]
        for i in range(len(conditions)):
            for key, value in stim_conditions_dict['Orientation'].items():
                if value == conditions[i][0] and stim_conditions_dict['y_range'][key] == conditions[i][1]:
                    onset_offset[i].append((stim_conditions_dict['Onset'][key], stim_conditions_dict['Offset'][key]))

        stimulus_df = combine_stim_pairs(condition_df, onset_offset) 
    
    # if type == "":
    #     conditions = 

    return stimulus_df

# def generate_random_stim(type):
#     '''
#     Generate random stimulus 
#     '''
#     if type == 'funkystim':
#         random_funkiness = random.randint(0,3)
#         random_orientation = random.randint(0,1)
#         random_contrast = random.randint(0,1)
        
#         return random_funkiness, random_orientation, random_contrast
    
#     if type == 'dg':
#         degrees = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
#         random_degree = random.choice(degrees)

#         return random_degree
    
#     if type == 'fs':
#         x_pos = [-55, -50, -45, -40, -35, -30, -25]
#         y_pos = [-30, -25, -20, -15, -10, -5, 0]

#         random_x = random.choice(x_pos)
#         random_y = random.choice(y_pos)

#         return random_x, random_y

#     if type == 'l':
#         speed = [10, 20, 30, 40]
#         random_speed = random.choice(speed)

#         return random_speed

#     if type == 'md':
#         direction = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
#         random_direction = random.choice(direction)

#         return random_direction

#     if type == 'ms':
#         orientation = [0, 90, 180, 270]
#         y_range = [-15, -10, -5, 0, 5, 10]

#         random_orientation = random.choice(orientation)
#         random_y_range = random.choice(y_range)

#         return random_orientation, random_y_range


# def pick_random_pair(stimulus_df, type, funkiness=None, orientation=None, contrast=None, degree=None, x_pos=None, y_pos=None, speed=None, direction=None, y_range=None):
#     '''
#     Given random stimulus, pick a random onset/offset pair
#     '''
#     if type == 'funkystim':
#         filtered_df = stimulus_df[(stimulus_df['Funkiness'] == funkiness) &
#                                         (stimulus_df['Orientation (V/H)'] == orientation) &
#                                         (stimulus_df['Contrast'] == contrast)]
#         pairs_list = filtered_df['Pairs'].iloc[0]
#         random_pair = random.choice(pairs_list)

#         print(f"Given the stimulus -------")
#         print(f"Funkiness: {funkiness}, Orientation: {orientation}, Contrast: {contrast}")
#         print(f"Random Onset/Offset frames to test: {random_pair}")

#     if type == 'dg':
#         filtered_df = stimulus_df[stimulus_df['Degrees'] == degree]
#         pairs_list = filtered_df['Pairs'].iloc[0]
#         random_pair = random.choice(pairs_list)

#         print(f"Given the stimulus -------")
#         print(f"{degree} deg")
#         print(f"Random Onset/Offset frames to test: {random_pair}")

#     if type == 'fs':
#         filtered_df = stimulus_df[(stimulus_df['x_pos'] == x_pos) & (stimulus_df['y_pos'] == y_pos)]
#         pairs_list = filtered_df['Pairs'].iloc[0]
#         random_pair = random.choice(pairs_list)

#         print(f"Given the stimulus -------")
#         print(f"x_pos: {x_pos}, y_pos: {y_pos}")
#         print(f"Random Onset/Offset frames to test: {random_pair}")
    
#     if type == 'l':
#         filtered_df = stimulus_df[stimulus_df['Speed'] == speed]
#         pairs_list = filtered_df['Pairs'].iloc[0]
#         random_pair = random.choice(pairs_list)

#         print(f"Given the stimulus -------")
#         print(f"{speed}")
#         print(f"Random Onset/Offset frames to test: {random_pair}")

#     if type == 'md':
#         filtered_df = stimulus_df[stimulus_df['Direction'] == direction]
#         pairs_list = filtered_df['Pairs'].iloc[0]
#         random_pair = random.choice(pairs_list)

#         print(f"Given the stimulus -------")
#         print(f"{direction}")
#         print(f"Random Onset/Offset frames to test: {random_pair}")

#     if type == 'ms':
#         filtered_df = stimulus_df[(stimulus_df['Orientation'] == orientation) & (stimulus_df['y_range'] == y_range)]
#         pairs_list = filtered_df['Pairs'].iloc[0]
#         random_pair = random.choice(pairs_list)

#         print(f"Given the stimulus -------")
#         print(f"Orientation: {orientation}, y_range: {y_range}")
#         print(f"Random Onset/Offset frames to test: {random_pair}")

#     return random_pair

# def sample_auc(stimulus_df, C, on_times, off_times, type, pre_stim_window=0, stim_extension=0):
#     if type == 'funkystim':
#         funkiness, orientation, contrast = generate_random_stim(type)
#         random_pair = pick_random_pair(stimulus_df, type, funkiness, orientation, contrast)

#     elif type == 'dg':
#         degree = generate_random_stim(type)
#         random_pair = pick_random_pair(stimulus_df, type, degree=degree)
    
#     elif type == 'fs':
#         x_pos, y_pos = generate_random_stim(type)
#         random_pair = pick_random_pair(stimulus_df, type, x_pos=x_pos, y_pos=y_pos)
    
#     elif type == 'l':
#         speed = generate_random_stim(type)
#         random_pair = pick_random_pair(stimulus_df, type, speed=speed)
    
#     elif type == 'md':
#         direction = generate_random_stim(type)
#         random_pair = pick_random_pair(stimulus_df, type, direction=direction)
    
#     elif type == 'ms':
#         orientation, y_range = generate_random_stim(type)
#         random_pair = pick_random_pair(stimulus_df, type, orientation=orientation, y_range=y_range)

#     stim_idx = on_times.index(random_pair[0])

#     # random_pair_auc = auc[:, stim_idx]
#     frame = [on_times[stim_idx], off_times[stim_idx]]
#     responses = C[:, frame[0]-pre_stim_window:frame[1]+stim_extension]
#     random_pair_auc = np.sum(responses, axis=1)
    
#     print(f"Stim Index {stim_idx} corresponds to the stim pair: {frame}")
#     print(f"AUC for above stimulus:\n {random_pair_auc}")

#     return random_pair_auc

def split_stim_df(stimulus_df, stim_parameter):
    param_dict = {}
    for param in stimulus_df[stim_parameter].unique():
        var_name = stim_parameter + '_{}'.format(param)
        param_dict[var_name] = stimulus_df.loc[stimulus_df[stim_parameter] == param, ['Pairs']]

    return param_dict

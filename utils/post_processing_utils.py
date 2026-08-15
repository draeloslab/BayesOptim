import os
import pandas as pd
import numpy as np

def read_outliers_csv(filepath):
    df = pd.read_csv(filepath)
    b_or_s = "b" if "burgess" in filepath else "s"
    data_dict = {}
    for column_header in df.columns:
        data_dict_key = column_header.strip(f" ({b_or_s})")
        data_dict[data_dict_key] = df[column_header].dropna().astype(int).tolist()
    if len(data_dict) == 1:
        data_dict = list(data_dict.values())[0]
    return data_dict

def read_global_log(path):
    with open(os.path.join(path, "global.log"), 'r') as file:
        global_log = file.readlines()
    return global_log

def count_num_calibration(global_log):
    num_calibration = len([line.strip("\n") for line in global_log if "Stimulus: Sin Drift Gratings" in line or "Stimulus: Flashing spot" in line])
    return num_calibration

def stash_calibration_XY(path, stimX, stimY):
    global_log = read_global_log(path)
    num_calibration = count_num_calibration(global_log)

    stimX = np.array(stimX[num_calibration:])
    stimX = np.delete(stimX, [4,5,7], axis=1)
    stimY = stimY[num_calibration:]
    return stimX, stimY

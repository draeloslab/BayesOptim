
import os
import yaml
import pickle
from datetime import datetime as dt 
import numpy as np

def save_results(config, param_file_path, results_dict):

    ## first check if there is an output folder, if not create one
    if not os.path.exists('output'):
        os.makedirs('output')
        print("------- Created 'output' directory -------")

    ## Creating date-specific output folder
    output_base_name = f'output_{dt.now().strftime("%m")}_{dt.now().strftime("%d")}'
    existing_folder = [d for d in os.listdir('output') if d.startswith(output_base_name)] # checking to see if the i-th folder already exists
    i = len(existing_folder) + 1

    output_folder = os.path.join('output', f"{output_base_name}_{i}")
    os.makedirs(output_folder, exist_ok=True)

    ## Copy the parameter YAML file to the output folder
    with open(param_file_path, "rb") as src, open(os.path.join(output_folder, os.path.basename(param_file_path)), "wb") as dst:
        dst.write(src.read())
        print(f'Copied {param_file_path} to {output_folder}')

    ## Save results_dict as a pickle file
    results_file = os.path.join(output_folder, 'results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results_dict, f)
    print(f'Saved results dictionary as pickle file in {output_folder}')

    return results_file
    
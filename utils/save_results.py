
import os
import yaml
from datetime import datetime as dt 

def save_results(param_file_path, results_dict):

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

    ## Save results_dict as a YAML file 
    results_file = os.path.join(output_folder, 'results.yaml')
    with open(results_file, 'w') as f:
        yaml.dump(results_dict, f, default_flow_style=False)
    print(f'Saved results dictionary as YAML file in {output_folder}')
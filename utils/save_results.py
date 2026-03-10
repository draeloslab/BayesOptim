
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

    with open(results_file, 'rb') as f:
        results_dict_pickle = pickle.load(f)
        online_grid1 = np.array(results_dict_pickle['f_all'][0][-1])
        print(f"online grid1: {online_grid1}")
         #FIX THIS 
        for dim1 in range(len(config.exs)):
            for dim2 in range(dim1+1, len(config.exs)):
                x_range = config.exs[dim1]
                y_range = config.exs[dim2]
                X, Y = np.meshgrid(x_range, y_range, indexing='ij')
                online_grid2 = online_grid1.reshape(X.shape).T
                print(f"online grid2: {online_grid2}")
        # print(f'sample x: {results_dict_pickle["sample_x"]}')
        # print(f'sample y: {results_dict_pickle["sample_y"]}')
        # print(f'f_all: {results_dict_pickle["f_all"]}')

#     # Preparing for checking online and offline fits. saving the same data. need to check this. 
#     #does not accommodate reruns well
#     
#     for k, v in results_dict_pickle.items():
#             print(f"{k}: type={type(v)}, length={len(v)}")   
#     for key, value in results_dict_pickle["sample_x"][0].items():
#         print(f"Neuron 0 Sample X {key}: {value}, length = {len(value)}, type = {type(value)}")
#     items = results_dict_pickle['sample_x'][0]['initial'][1]
#     print(f"initial: {items}")
    


# #SAMPLE X

#     if len(results_dict_pickle['sample_x'][0]['initial']) > 1:
#         xarray1 = np.array(results_dict_pickle['sample_x'][1]['initial'][-1])
#         xarray2 = np.array(results_dict_pickle['sample_x'][1]['selected'])
#         new_xarray = np.vstack([xarray1, xarray2])
#         #print(f"new x shape{new_xarray.shape}")
#     else:
        # xarray1 = np.array(results_dict_pickle['sample_x'][0]['initial'])
        # xarray2 = np.array(results_dict_pickle['sample_x'][0]['selected'])
        # new_xarray = np.vstack([xarray1, xarray2])
        #print(f"new x shape{new_xarray.shape}")

# #SAMPLE Y

#     for key, value in results_dict_pickle["sample_y"][0].items():
#         print(f"Neuron 0 Sample Y {key}: value {value} length = {len(value)}, type = {type(value)}")

#     if len(results_dict_pickle['sample_y'][1]['initial']) > 1:
#         yarray1 = np.array(results_dict_pickle['sample_y'][1]['initial'][-1])
#         yarray2 = np.array(results_dict_pickle['sample_y'][1]['selected'])
#         new_yarray = np.concatenate([yarray1, yarray2])
#         #print(f"new y shape{new_yarray.shape}")
        
#     else: 
        # yarray1 = np.array(results_dict_pickle['sample_y'][0]['initial'])
        # yarray2 = np.array(results_dict_pickle['sample_y'][0]['selected'])
        # new_yarray = np.concatenate([yarray1, yarray2])
        #print(f"new y shape{new_yarray.shape}")

        return online_grid2 #new_xarray, new_yarray


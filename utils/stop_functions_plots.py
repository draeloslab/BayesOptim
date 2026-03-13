import os
import pickle
import matplotlib.pyplot as plt
import yaml
import numpy as np
from datetime import datetime as dt

def stop_functions_plots(section_key='stopping_allN'):
    """
    Generate EI trajectory plots for all neurons and hyperparameter settings across multiple experiment runs.

    Parameters:
        section_key (str): The results dictionary key to extract stopping metrics from each pickle file.
                        Defaults to 'stopping_allN'.

    Operation:
        - Iterates over all experiment folders with prefix 'output_' in the 'output' directory.
        - Loads hyperparameter values ('gamma', 'nu', neuron count) from 'parameters_indep.yml'.
        - Loads stopping metrics from 'results.pkl' under the specified section_key.
        - For each run, plots EI trajectory (optimization steps vs EI) for each neuron, with:
            - Color distinguishing neurons,
            - Line style distinguishing nu or gamma,
            - Manual legends for clear labeling.
        - Displays two plots: one focused on nu values, one on gamma values.
        - Prints warnings if files or keys are missing.
        #note that only 4 linestyles exist presently, could be increased to accommodate more nu/gamma comparisons

    Output:
        - Displays matplotlib plots of EI trajectories with custom legends.
    """
    
    output_dir = 'output'
    folders = [os.path.join(output_dir, d) for d in os.listdir(output_dir)
               if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('output_')]
    total_stopping_allN= []
    folder_labels = []
    gammas = []
    nus = []

    for idx, folder in enumerate(folders):
        results_file = os.path.join(folder, 'results.pkl')
        parameters_file = os.path.join(folder, 'parameters_indep.yml')
        if os.path.exists(parameters_file):
            with open(parameters_file, 'r') as f:
                params = yaml.safe_load(f)
                gamma = params['Optimizer']['gamma']
                nu = params['Optimizer']['nu']
                N = params['Neurons']['N']
                stopping_crit = params['Optimizer']['optim_1']['stopping_crit']
                gammas.append(gamma)
                nus.append(nu)
        if os.path.exists(results_file):
            with open(results_file, 'rb') as f:
                results_dict = pickle.load(f)
            if section_key in results_dict:
                total_stopping_allN.append(results_dict[section_key])  #collect Stopping_all_N from results_dict
                folder_labels.append(os.path.basename(folder))
            else:
                print(f"Key '{section_key}' not found in {results_file}")
        else:
            print(f"Path does not exist")

    # Define line styles and colors
    line_styles = ['-', '--', '-.', ':']  
    colors = plt.cm.tab10.colors  
    plt.figure(figsize=(10, 6))
    for run_idx, stopping_allN in enumerate(total_stopping_allN):
        # Select line style for this nu
        line_style = line_styles[run_idx % len(line_styles)]

        for neuron_idx, neuron_vals in enumerate(stopping_allN):
            ei_traj = [val[0] for val in neuron_vals] #val[1] = PI if of interest
            color = colors[neuron_idx % len(colors)] 
            plt.plot(
                ei_traj,
                linestyle=line_style,
                color=color,
                label=f'Neuron {neuron_idx}, nu = {nu}' if run_idx == 0 else None  
            )
    # Construct legend manually to show neuron and nu values
    from matplotlib.lines import Line2D
    neuron_labels = [f'Neuron {i+1}' for i in range(len(stopping_allN))]
    neuron_lines = [Line2D([0], [0], color=colors[i % len(colors)], lw=2) for i in range(len(stopping_allN))]

    nu_labels = [f'Nu {nu}' for nu in nus]
    nu_lines = [Line2D([0], [0], color='k', linestyle=line_styles[i % len(line_styles)], lw=2) for i in range(len(stopping_allN))]

    plt.xlabel('Optimization Step')
    plt.ylabel('Expected Improvement (EI)')
    plt.title('EI Trajectory for All Neurons and Nu Settings')

    # Add both legends
    plt.legend(neuron_lines + nu_lines, neuron_labels + nu_labels, loc='best', fontsize='small')

    plt.tight_layout()
    plt.show()

    for run_idx, stopping_allN in enumerate(total_stopping_allN):
            # Select line style for this gamma
            line_style = line_styles[run_idx % len(line_styles)]

            for neuron_idx, neuron_vals in enumerate(stopping_allN):
                ei_traj = [val[0] for val in neuron_vals] #val[1] = PI in case it is of interest
                color = colors[neuron_idx % len(colors)] 
                plt.plot(
                    ei_traj,
                    linestyle=line_style,
                    color=color,
                    label=f'Neuron {neuron_idx}, gamma = {gamma}' if run_idx == 0 else None  
                )

    # Construct legend manually to show neuron and gamma values
    neuron_labels = [f'Neuron {i+1}' for i in range(len(stopping_allN))]
    neuron_lines = [Line2D([0], [0], color=colors[i % len(colors)], lw=2) for i in range(len(stopping_allN))]

    gamma_labels = [f'Gamma {gamma}' for gamma in gammas]
    gamma_lines = [Line2D([0], [0], color='k', linestyle=line_styles[i % len(line_styles)], lw=2) for i in range(len(stopping_allN))]

    
    plt.xlabel('Optimization Step')
    plt.ylabel('Expected Improvement (EI)')
    plt.title('EI Trajectory for All Neurons and Nu Settings')

    # Add both legends
    plt.legend(neuron_lines + gamma_lines, neuron_labels + gamma_labels, loc='best', fontsize='small')

    plt.tight_layout()
    plt.show()

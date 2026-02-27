import os
import pickle
import matplotlib.pyplot as plt
import yaml
import numpy as np
from datetime import datetime as dt

def stop_functions_plots(section_key='stopping_allN'):
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
        # Select line style for this gamma
        line_style = line_styles[run_idx % len(line_styles)]

        for neuron_idx, neuron_vals in enumerate(stopping_allN):
            ei_traj = [val[0] for val in neuron_vals] #val[1] = PI instead of EI
            #print(f'neuron {neuron_idx}, ei_traj {ei_traj}')
            color = colors[neuron_idx % len(colors)] 
            plt.plot(
                ei_traj,
                linestyle=line_style,
                color=color,
                label=f'Neuron {neuron_idx}, nu = {nu}' if run_idx == 0 else None  #add gamma if that's what you change across runs Only add neuron label once
            )

    # Construct legend manually to show neurons and gamma styles
    from matplotlib.lines import Line2D
    neuron_labels = [f'Neuron {i+1}' for i in range(len(stopping_allN))]
    neuron_lines = [Line2D([0], [0], color=colors[i % len(colors)], lw=2) for i in range(len(stopping_allN))]

    # gamma_labels = [f'Gamma {gamma}' for gamma in gammas]
    # gamma_lines = [Line2D([0], [0], color='k', linestyle=line_styles[i % len(line_styles)], lw=2) for i in range(len(stopping_allN))]

    nu_labels = [f'Nu {nu}' for nu in nus]
    nu_lines = [Line2D([0], [0], color='k', linestyle=line_styles[i % len(line_styles)], lw=2) for i in range(len(stopping_allN))]

    plt.xlabel('Optimization Step')
    plt.ylabel('Expected Improvement (EI)')
    plt.title('EI Trajectory for All Neurons and Nu Settings')

    # Add both legends
    plt.legend(neuron_lines + nu_lines, neuron_labels + nu_labels, loc='best', fontsize='small')

    plt.tight_layout()
    plt.show()

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

    for idx, folder in enumerate(folders):
        results_file = os.path.join(folder, 'results.pkl')
        parameters_file = os.path.join(folder, 'parameters_indep.yml')
        if os.path.exists(parameters_file):
            with open(parameters_file, 'r') as f:
                params = yaml.safe_load(f)
                gamma = params['Optimizer']['gamma']
                N = params['Neurons']['N']
                stopping_crit = params['Optimizer']['optim_1']['stopping_crit']
                gammas.append(gamma)
        if os.path.exists(results_file):
            with open(results_file, 'rb') as f:
                results_dict = pickle.load(f)
            if section_key in results_dict:
                total_stopping_allN.append(results_dict[section_key])
                folder_labels.append(os.path.basename(folder))
            else:
                print(f"Key '{section_key}' not found in {results_file}")
        else:
            print(f"Path does not exist")

    # Define line styles and colors
    line_styles = ['-', '--', '-.', ':']  
    colors = plt.cm.tab10.colors  
    # Example: results_dicts = [results_dict_1, results_dict_2] where each has a 'gamma' key and 'stopping_allN' value
    plt.figure(figsize=(10, 6))
    for run_idx, stopping_allN in enumerate(total_stopping_allN):
        # Select line style for this gamma
        line_style = line_styles[run_idx % len(line_styles)]

        for neuron_idx, neuron_vals in enumerate(stopping_allN):
            ei_traj = [val[0] for val in neuron_vals]
            print(f'neuron {neuron_idx}, ei_traj {ei_traj}')
            color = colors[neuron_idx % len(colors)]  # Wrap around if more neurons than colors
            plt.plot(
                ei_traj,
                linestyle=line_style,
                color=color,
                label=f'Neuron {neuron_idx+1}, gamma={gamma}' if run_idx == 0 else None  # Only add neuron label once
            )

    # Construct legend manually to show neurons and gamma styles
    from matplotlib.lines import Line2D
    neuron_labels = [f'Neuron {i+1}' for i in range(len(stopping_allN))]
    neuron_lines = [Line2D([0], [0], color=colors[i % len(colors)], lw=2) for i in range(len(stopping_allN))]

    gamma_labels = [f'Gamma {gamma}' for gamma in gammas]
    gamma_lines = [Line2D([0], [0], color='k', linestyle=line_styles[i % len(line_styles)], lw=2) for i in range(len(stopping_allN))]

    plt.xlabel('Optimization Step')
    plt.ylabel('Expected Improvement (EI)')
    plt.title('EI Trajectory for All Neurons and Gamma Settings')

    # Add both legends
    plt.legend(neuron_lines + gamma_lines, neuron_labels + gamma_labels, loc='best', fontsize='small')

    plt.tight_layout()
    plt.show()

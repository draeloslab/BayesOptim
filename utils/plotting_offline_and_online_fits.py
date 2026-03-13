import matplotlib.pylab as plt
import scipy.stats as stats
import numpy as np
from model.optimizer import calc_offline_fit 
import os
import yaml
import pickle
from datetime import datetime as dt 
import numpy as np

def gathering_data(config, results_file, neuron_idx): 
    with open(results_file, 'rb') as f:
            results_dict_pickle = pickle.load(f)
            online_grid1 = np.array(results_dict_pickle['f_all'][neuron_idx][-1])
            print(f"online grid1: {online_grid1}")
            #FIX THIS 
            for dim1 in range(len(config.exs)):
                for dim2 in range(dim1+1, len(config.exs)):
                    x_range = config.exs[dim1]
                    y_range = config.exs[dim2]
                    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
                    online_grid2 = online_grid1.reshape(X.shape)
                    print(f"online grid2: {online_grid2}")
        # print(f'sample x Neuron 0: {results_dict_pickle["sample_x"][0]["initial"]}')
        # print(f'sample x: {results_dict_pickle["sample_x"][0]["selected"]}')
        # print(f'sample y: {results_dict_pickle["sample_y"][0]["initial"]}')
        # print(f'sample y: {results_dict_pickle["sample_y"][0]["selected"]}')
        # print(f'sample x Neuron 1: {results_dict_pickle["sample_x"][1]["initial"]}')
        # print(f'sample x: {results_dict_pickle["sample_x"][1]["selected"]}')
        # print(f'sample y: {results_dict_pickle["sample_y"][1]["initial"]}')
        # print(f'sample y: {results_dict_pickle["sample_y"][1]["selected"]}')

# #SAMPLE X

            xarray1 = np.array(results_dict_pickle['sample_x'][neuron_idx]['initial'][-1])
            xarray1_flat = xarray1.reshape(-1, xarray1.shape[-1])

            xarray2_list = results_dict_pickle['sample_x'][neuron_idx]['selected']
            xarray2 = np.vstack(xarray2_list)

            new_xarray = np.vstack([xarray1_flat, xarray2])
            print(f"new_xarray {new_xarray} new x shape: {new_xarray.shape}")

# #SAMPLE Y
            yarray1 = np.array(results_dict_pickle['sample_y'][neuron_idx]['initial'][-1])
            yarray1_flat = yarray1.reshape(-1)
            yarray2 = np.array(results_dict_pickle['sample_y'][neuron_idx]['selected']).reshape(-1)
            print('Shape yarray1_flat:', yarray1_flat.shape)
            print('Shape yarray2:', yarray2.shape)

            new_yarray = np.concatenate([yarray1_flat, yarray2])
            print(f"new y array {new_yarray} new y shape: {new_yarray.shape}")
            return online_grid2, new_xarray, new_yarray
    

def plotting_offline_and_online_fits(config, f, online_grid2, neuron_idx):
    for dim1 in range(len(config.exs)):
            for dim2 in range(dim1+1, len(config.exs)):
                x_range = config.exs[dim1]
                y_range = config.exs[dim2]
                X, Y = np.meshgrid(x_range, y_range, indexing='ij')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Neuron {neuron_idx} Offline vs Online GP Fit")

    offline_peak = config.x_star[np.argmax(f)]
    online_peak  = config.x_star[np.argmax(online_grid2)]

    # --- Offline ---
    cf0 = axes[0].contourf(
        f.reshape(X.shape).T,
        extent=(x_range[0]-0.5, x_range[-1]+0.5, y_range[0]-0.5, y_range[-1]+0.5),
        origin="lower", cmap="viridis"
    )
    axes[0].set_title("Offline GP Fit")

    axes[0].plot(offline_peak[0], offline_peak[1], "yo",
                label=f"GP offline peak: ({offline_peak[0]:.2f}, {offline_peak[1]:.2f})")

    fig.colorbar(cf0, ax=axes[0], label="GP Mean")
    axes[0].legend(loc="best")

    # --- Online ---
    cf1 = axes[1].contourf(
        online_grid2.T,
        extent=(x_range[0]-0.5, x_range[-1]+0.5, y_range[0]-0.5, y_range[-1]+0.5),
        origin="lower", cmap="viridis"
    )
    axes[1].set_title("Online GP Fit")

    axes[1].plot(online_peak[0], online_peak[1], "yo",
                label=f"GP online peak: ({online_peak[0]:.2f}, {online_peak[1]:.2f})")

    fig.colorbar(cf1, ax=axes[1], label="GP Mean")
    axes[1].legend(loc="best")

    plt.tight_layout()
    plt.show()

    print(f"offline {f.reshape(X.shape)}")
    print(f"online {online_grid2}")

    np.allclose(f.reshape(X.shape), online_grid2, atol = 1e-3)

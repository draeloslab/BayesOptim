import matplotlib.pylab as plt
import numpy as np
import pickle

def gathering_data(config, results_file, neuron_idx): 
    """
    Load saved Bayesian optimization results and assemble data needed for plotting offline and online GP fits

    This helper function reads a pickled results dictionary and returns:
      1) The final “online” GP mean surface for a given neuron (reshaped to the 2D stimulus grid),
      2) The full set of sampled X locations used to train the GP (initial design + sequentially selected),
      3) The corresponding observed Y values (initial responses + sequentially selected responses).

    Parameters
    ----------
    config : object
        Configuration object providing stimulus axes:
          - config.exs : list of 1D arrays
              Stimulus axis values for each dimension. This function currently assumes you want to
              reshape to a 2D grid using one (dim1, dim2) pair from config.exs.

    results_file : From Bayesopt results_dict
        
          - 'f_all'[neuron_idx][-1] : array-like
              Final GP mean evaluated on the full candidate set 
          - 'sample_x'[neuron_idx]['initial'] : list 
          - 'sample_x'[neuron_idx]['selected'] : list
          - 'sample_y'[neuron_idx]['initial'] : list
          - 'sample_y'[neuron_idx]['selected'] : list

    neuron_idx : int
        Index of the neuron whose results should be extracted.

    Returns
    -------
    online_grid2 : numpy.ndarray
        The final online GP mean surface reshaped to the 2D plotting grid, shape (n_x, n_y),
        where n_x = len(config.exs[dim1]) and n_y = len(config.exs[dim2]).
    new_xarray : numpy.ndarray
        All sampled X locations used for GP training, stacked as:
        [initial_X; selected_X], shape (n_samples_total, d).

    new_yarray : numpy.ndarray
        All sampled Y observations aligned with new_xarray, stacked as:
        [initial_y; selected_y], shape (n_samples_total,).
    """
    with open(results_file, 'rb') as f:
            results_dict_pickle = pickle.load(f)
            online_grid1 = np.array(results_dict_pickle['f_all'][neuron_idx][-1])
            print(f"online grid1: {online_grid1}")
            for dim1 in range(len(config.exs)):
                for dim2 in range(dim1+1, len(config.exs)):
                    x_range = config.exs[dim1]
                    y_range = config.exs[dim2]
                    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
                    online_grid2 = online_grid1.reshape(X.shape)
                    print(f"online grid2: {online_grid2}")
 
            #SAMPLE X
            xarray1 = np.array(results_dict_pickle['sample_x'][neuron_idx]['initial'][-1])
            xarray1_flat = xarray1.reshape(-1, xarray1.shape[-1])

            xarray2_list = results_dict_pickle['sample_x'][neuron_idx]['selected']
            xarray2 = np.vstack(xarray2_list)

            new_xarray = np.vstack([xarray1_flat, xarray2])
            print(f"new_xarray {new_xarray} new x shape: {new_xarray.shape}")

            #SAMPLE Y
            yarray1 = np.array(results_dict_pickle['sample_y'][neuron_idx]['initial'][-1])
            yarray1_flat = yarray1.reshape(-1)
            yarray2 = np.array(results_dict_pickle['sample_y'][neuron_idx]['selected']).reshape(-1)
            print('Shape yarray1_flat:', yarray1_flat.shape)
            print('Shape yarray2:', yarray2.shape)

            new_yarray = np.concatenate([yarray1_flat, yarray2])
            print(f"new y array {new_yarray} new y shape: {new_yarray.shape}")
            return online_grid2, new_xarray, new_yarray
    

def plotting_offline_and_online_fits(config, f, online_grid2, neuron_idx):
    """
    Plot side-by-side offline vs online GP mean surfaces for a given neuron, and mark their peaks.

    The “offline” surface is provided as a flat vector `f` (GP mean over the full candidate set),
    which is reshaped to the 2D stimulus grid. The “online” surface is provided as `online_grid2`
    already reshaped to the same 2D grid. The function plots both with contourf and overlays the
    argmax (“peak”) location in stimulus coordinates.

    Parameters
    ----------
    config : object
        Configuration object providing stimulus axes and stimulus coordinates:
    f : array-like
        Offline GP mean evaluated over the full candidate set 

    online_grid2 : numpy.ndarray
        Online GP mean surface already reshaped to the 2D plotting grid, shape (n_x, n_y).

    neuron_idx : int
        Neuron index (used for labeling/titles).

    Returns
    -------
    None
        Displays a matplotlib figure with two subplots and prints the reshaped arrays.
        Also computes (but does not return) an np.allclose comparison of offline vs online grids.
    """
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

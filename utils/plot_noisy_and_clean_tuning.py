import matplotlib.pylab as plt
import numpy as np

def plot_noisy_and_clean_tuning(config, neuron_idx): 
    """
    Plot side-by-side noise-free (analytic/precomputed) and noisy DoG tuning surfaces for one neuron.

    This function visualizes, for a given neuron, (1) the noise-free Difference-of-Gaussians (DoG)
    response surface that was precomputed during tuning curve generation (stored in
    ``config.SimPop.responses``), and (2) a noisy DoG response surface obtained by evaluating the
    neuron's response at every stimulus point with add_noise=True in the sample method of Auditory neurons
    The function also marks the peak for each surface:
    Parameters
    ----------
    config : object
    neuron_idx : int
        Index of the neuron to plot.

    Returns
    -------
    None
        Displays a matplotlib figure with two subplots.
    """
    SimPop = config.SimPop
    for dim1 in range(len(config.exs)):
            for dim2 in range(dim1+1, len(config.exs)):
                x_range = config.exs[dim1]
                y_range = config.exs[dim2]
                X, Y = np.meshgrid(x_range, y_range, indexing='ij')

    import matplotlib.pyplot as plt
    noisy_grid = []
    for xi in SimPop.x_star:  # xi is an input stimulus point (shape [d])
        z = SimPop.sample(xi, add_noise = True)      # z is a vector of responses: shape [N]
        noisy_grid.append(z[neuron_idx])    # get the response of neuron 1 (or whichever neuron you want)

    # Convert to numpy array
    noisy_grid = np.array(noisy_grid)
    noisy_peak = config.x_star[np.argmax(noisy_grid)]
    noisy_grid_reshaped = noisy_grid.reshape(X.shape).T

    noNoise_peak  = config.x_star[np.argmax(SimPop.responses[neuron_idx])]
    no_noise_grid = SimPop.responses[neuron_idx].reshape(X.shape).T


    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Neuron {neuron_idx}")

    # Noise_free response
    c0 = axs[0].contourf(no_noise_grid, extent=(x_range[0] - 0.5, x_range[-1] + 0.5, y_range[0]-0.5, y_range[-1]+0.5),
                                origin='lower', cmap='viridis')
    axs[0].set_title("Noise-Free DoG Response Neuron")
    axs[0].plot(noNoise_peak[0], noNoise_peak[1], "yo",
                label=f"Noise-free peak: ({noNoise_peak[0]:.2f}, {noNoise_peak[1]:.2f})")

    fig.colorbar(c0, ax=axs[0])
    axs[0].legend(loc="best")

    # Noisy response
    c1 = axs[1].contourf(noisy_grid_reshaped, extent=(x_range[0] - 0.5, x_range[-1] + 0.5, y_range[0]-0.5, y_range[-1]+0.5),
                                origin='lower', cmap='viridis')
    axs[1].set_title("Noisy DoG Response")
    axs[1].plot(noisy_peak[0], noisy_peak[1], "yo",
                label=f"Noisy peak: ({noisy_peak[0]:.2f}, {noisy_peak[1]:.2f})")
    fig.colorbar(c1, ax=axs[1])
    axs[1].legend(loc="best")

    plt.tight_layout()
    plt.show()



        


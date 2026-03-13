import matplotlib.pylab as plt
import scipy.stats as stats
import numpy as np
from sklearn.metrics import mean_squared_error

def sampling_for_plots_auditory(config): 
    ''' 
    Sample neural responses for the entire sample space

    params:
    ---------
    neuron_num (int): index of a specific neuron
    config : configuration object, containing SimPop object

    returns:
    ---------
    numpy array

    '''
    np.random.seed(config.params['General']['seed'])
    N = config.N
    d = config.d 
    SimPop = config.SimPop

    for dim1 in range(len(config.exs)):
        for dim2 in range(dim1+1, len(config.exs)):
            X, Y= np.meshgrid(config.exs[dim1], config.exs[dim2], indexing = 'ij') 
            pos = np.dstack((X, Y))  
            Z = np.zeros((N, len(config.exs[dim1]) * len(config.exs[dim2])))

    #sample the whole stimulus space
    for n in range(N):
        for i in range(len(config.exs[dim1])): 
            for j in range(len(config.exs[dim2])): 
                resp = config.SimPop.sample(pos[i,j])[n]
                Z[n][i*len(config.exs[dim2])+ j] = resp  #filling up the whole grid
    return Z
def plot_tuningcurves_sampled_auditory(config):
    ''' 
    Plot tuning curves as 2D imshow plot for each dimension pair with true, sampled, and offline peaks

    params:
    ---------
    neuron_num (int): index of a specific neuron
    config : configuration object, containing SimPop object
    f_peak (tuple): offline peak locations (optional)

    returns:
    ---------
    None 

    '''
    Z = sampling_for_plots_auditory(config)
    #unravel_index(Z.argmax(), np.transpose(Z.shape))
    for dim1 in range(len(config.exs)):
        for dim2 in range(dim1+1, len(config.exs)):
            for n in range(config.N):
                plt.figure(figsize=(8, 6))
                x_range = config.exs[dim1]
                y_range = config.exs[dim2]
                X, Y = np.meshgrid(x_range, y_range, indexing = 'ij')
                sampled_peak = config.x_star[np.argmax(Z[n])]
                Z_reshaped = Z[n].reshape(X.shape).T
                #print(f"Z_reshaped neuron {n}: {Z_reshaped}; Z_reshaped neuron: {Z_reshaped.shape}; Z shape is: {Z.shape}")
                plt.contourf(Z_reshaped, extent=(x_range[0] - 0.5, x_range[-1] + 0.5, y_range[0]-0.5, y_range[-1]+0.5),
                            origin='lower', cmap='viridis')
                plt.plot(config.SimPop.peaks[n][0], config.SimPop.peaks[n][1], 'ro',
                        label = f"true peak: ({config.SimPop.peaks[n][0]:.2f}, {config.SimPop.peaks[n][1]:.2f})")
                plt.plot(sampled_peak[dim1], sampled_peak[dim2], 'bo',
                        label = f"sampled peak: ({sampled_peak[dim1]:.2f}, {sampled_peak[dim2]:.2f})")
                #if f_peak is not None:
                #     plt.plot(f_peak[dim1], f_peak[dim2], 'yo', 
                #             label = f"GP offline peak: ({f_peak[dim1]:.2f}, {f_peak[dim2]:.2f})")
                plt.legend()
                plt.xlabel(f'Dimension {dim2 + 1}')
                plt.ylabel(f'Dimension {dim1 + 1}')
                plt.xticks(y_range)
                plt.yticks(x_range)
                plt.colorbar(label='Sampled Response')
                plt.title(f'Neuron {n} - Tuning Curves for Dimension {dim1 + 1} vs. Dimension {dim2 + 1}')
                plt.tight_layout()
                plt.show()


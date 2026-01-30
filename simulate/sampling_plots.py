import matplotlib.pylab as plt
import scipy.stats as stats
import numpy as np

# class sampling_plots():
#     def __init__(self):
#         self.means= means
#         self.covs=covs
def sampling_for_plots_Penny(neuron_num, config):
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
    for dim1 in range(len(config.exs)):
        for dim2 in range(dim1+1, len(config.exs)):
            X, Y= np.meshgrid(config.exs[dim1], config.exs[dim2], indexing = 'ij') 
            pos = np.dstack((X, Y))  # 16, 10, 2
            Z = np.zeros((len(config.exs[dim2])*len(config.exs[dim1])))
            sampled_points = []
            for i in range(len(config.exs[dim1])):  # 16
                for j in range(len(config.exs[dim2])):  # 10
                    # putting the set seed here kind of lose the sampling power??
                    # np.random.seed(config.params['General']['seed'])
                    # print(pos[i,j])
                    resp = config.SimPop.sample(pos[i,j])[neuron_num]
                    stim_point = pos[i, j]
                    if neuron_num == 1:
                        sampled_points.append(stim_point)
                    # print(resp)
                    Z[i*len(config.exs[dim2])+ j] = resp # just sampling
            #print(f"sampled neuron : {sampled_points}")
            print(f"Z_gridsample neuron {neuron_num}: {Z}")
            # # sample for nD array; discarded
            # Z = np.zeros((len(exs[dim2]), len(exs[dim1])))
            # for i in range(Z.shape[dim1]):  # 10
            #     for j in range(Z.shape[dim2]):  # 16
            #         Z[i,j] = SimPop.sample(pos[i,j])[neuron_num]  # sample from the population
    return Z
def plot_tuningcurves_sampled_Penny(neuron_num, config, f_peak = None):
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
    Z = sampling_for_plots_Penny(neuron_num, config)
    sampled_peak = config.x_star[np.argmax(Z)] #unravel_index(Z.argmax(), np.transpose(Z.shape))
    for dim1 in range(len(config.exs)):
        for dim2 in range(dim1+1, len(config.exs)):
            for n in range(neuron_num, neuron_num+1):
                plt.figure(figsize=(8, 6))
                x_range = config.exs[dim1]
                y_range = config.exs[dim2]
                X, Y = np.meshgrid(x_range, y_range, indexing = 'ij')
                Z_reshaped = Z.reshape(X.shape)#.T
                print(f"Z_reshaped neuron {neuron_num}: {Z_reshaped}; Z_reshaped neuron: {Z_reshaped.shape}; Z shape is: {Z.shape}")
                plt.imshow(Z_reshaped, #extent=(x_range[0] - 0.5, x_range[-1] + 0.5, y_range[0]-0.5, y_range[-1]+0.5),
                            origin='lower', cmap='viridis', aspect='auto')
                plt.plot(config.SimPop.peaks[neuron_num][0], config.SimPop.peaks[neuron_num][1], 'ro',
                        label = f"true peak: ({config.SimPop.peaks[neuron_num][0]:.2f}, {config.SimPop.peaks[neuron_num][1]:.2f})")
                plt.plot(sampled_peak[dim1], sampled_peak[dim2], 'bo',
                        label = f"sampled peak: ({sampled_peak[dim1]:.2f}, {sampled_peak[dim2]:.2f})")
                if f_peak is not None:
                    plt.plot(f_peak[dim1], f_peak[dim2], 'yo', 
                            label = f"GP offline peak: ({f_peak[dim1]:.2f}, {f_peak[dim2]:.2f})")
                plt.legend()
                plt.xlabel(f'Dimension {dim2 + 1}')
                plt.ylabel(f'Dimension {dim1 + 1}')
                plt.xticks(y_range)
                plt.yticks(x_range)
                plt.colorbar(label='Sampled Response')
                plt.title(f'Neuron {neuron_num} - Tuning Curves for Dimension {dim1 + 1} vs. Dimension {dim2 + 1}')
                plt.show()


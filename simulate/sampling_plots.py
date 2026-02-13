import matplotlib.pylab as plt
import scipy.stats as stats
import numpy as np
from simulate.ground_truth_plot import plot_tuningcurves_Penny
from sklearn.metrics import mean_squared_error

# class sampling_plots():
#     def __init__(self):
#         self.means= means
#         self.covs=covs
def sampling_for_plots_Penny(neuron_num, config): #formerly included neuron number as input
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
    correct_neurons = 0
    N = config.N
    SimPop = config.SimPop
    Z_dog = plot_tuningcurves_Penny(N,SimPop,config) 
    print(f"config_exs: {config.exs}")
    #max_tests = config.params['General']['max_tests']

    for dim1 in range(len(config.exs)):
        for dim2 in range(dim1+1, len(config.exs)):
            X, Y= np.meshgrid(config.exs[dim1], config.exs[dim2], indexing = 'ij') 
            pos = np.dstack((X, Y))  
            Z = np.zeros((N, len(config.exs[dim1]) * len(config.exs[dim2])))
            print(f"Z shape: {Z.shape}")

    gsPr_list = []
    correct_neurons = 0
    #tests_so_far = 0

    for n in range(N):
        flagged = False
        sampled_responses = []
        num_points = len(config.exs[dim1]) * len(config.exs[dim2])
        half_points = 15
        #all_indices = [(i, j) for i in range(len(config.exs[dim1])) for j in range(len(config.exs[dim2]))]
        row_indices = np.arange(len(config.exs[dim1]))
        random_rows = np.random.choice(row_indices, size=len(config.exs[dim1])//2, replace=False)
        #print(f"random rows {random_rows}")
        indices_struct = [(i, j) for i in random_rows for j in range(len(config.exs[dim2]))]
        np.random.shuffle(indices_struct)
        #print(f"indices_struct: {indices_struct}")
        #sample_indices = np.random.choice(range(num_points), size= 10, replace=False)
        #sample_indices_ij = [all_indices[idx] for idx in sample_indices]
        #for idx, (i, j) in enumerate(sample_indices_ij):
        for (i, j) in indices_struct[:half_points]:
            #resp = config.SimPop.sample(pos[i, j])[n]
            #sampled_responses.append(resp)
            #max_sample_idx = np.argmax(sampled_responses)
            #peak_coords_pred = sample_indices_ij[max_sample_idx]
            peak_coords_pred = (i,j)
            # print(f"neuron {n} peak_coords_pred: {[peak_coords_pred]}")
            peak_coords_true = np.unravel_index(np.argmax(Z_dog), Z_dog.shape)
            mse = mean_squared_error(np.array(peak_coords_pred), np.array(peak_coords_true))
            
            if mse < .05 and not flagged:  #so it doesn't mark the same neuron correct twice
                correct_neurons += 1
                flagged = True
                break

            # Append running probability 
            gsPr_list.append(correct_neurons /N)

    # print(f"gsPr_list: {gsPr_list}")
    
    # for n in range(N):
    #     gsPr_list.append((correct_neurons)/N)
    #     sampled_responses = []
    #     num_points = len(config.exs[dim1]) * len(config.exs[dim2])
    #     #print(f"num_points: {num_points}")
    #     half_points = 15 #um_points // 2
    #     all_indices = [(i, j) for i in range(len(config.exs[dim1])) for j in range(len(config.exs[dim2]))]
    #     sample_indices = np.random.choice(range(num_points), size=half_points, replace=False)
    #     sample_indices_ij = [all_indices[idx] for idx in sample_indices]
    #     #print(f"sample_inices_ij {sample_indices_ij}")
    #     max_idx = np.argmax(sample_indices)
    #     print(f"max_idx {max_idx}")
    #     for i in range(len(config.exs[dim1])): 
    #         for j in range(len(config.exs[dim2])): 
    #             resp = config.SimPop.sample(pos[i,j])[n]
    #             Z[n][i*len(config.exs[dim2])+ j] = resp  #filling up the whole grid
    #     for (i, j) in sample_indices_ij: 
    #             resp = config.SimPop.sample(pos[i,j])[n]
    #             sampled_responses.append(resp)   #every other point is sampled
    #              # Find (i, j) where maximum response occurs among sampled points
    #             max_sample_idx = np.argmax(sampled_responses)
    #             #print(f"max_sample_idx: {max_sample_idx}")
    #             peak_coords_pred = sample_indices_ij[max_sample_idx]
    #             print(f"peak_coords_pred {peak_coords_pred}")
    #             peak_coords_true = np.unravel_index(np.argmax(Z_dog), Z_dog.shape)
    #             #print(f"peak_coords_true {peak_coords_true}")
    #             mse = mean_squared_error(np.array(peak_coords_pred), np.array(peak_coords_true))
    #             print(f"neuron {n} gs mse: {mse}")
    #     # print("stim_pred:", np.array(peak_coords_pred), "stim_true:", np.array(peak_coords_true))
    #     # print("peak_coords_pred:", peak_coords_pred, "peak_coords_true:", peak_coords_true)
    #     # print("mse input shape:", np.array(peak_coords_pred).shape, np.array(peak_coords_true).shape)
    #             if mse < .05:  
    #                 correct_neurons += 1
    #             #print(f"Z_max_neuron{n}: {Z[n][np.argmax(Z)]}")
    #     if n == N-1:
    #         gsPr_list.append(float(correct_neurons)/float(N))
    # print(f"gsPr_list: {gsPr_list}")
    # for entry in gsPr_list:
    #     print(type(entry), entry)
    
    # plt.plot(np.arange(0,len(gsPr_list)), gsPr_list, linestyle='-', color='r',label="Grid Sampling")
    # plt.xlabel('# of predictions')
    # plt.ylabel('Probability')
    # plt.title(f'Probability of making correct predictions')
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # textstr = f" # of neurons = {N}"
    # plt.text(0.50, 0.90, textstr,  fontsize=14, verticalalignment='top', bbox=props)
    
    # plt.legend()
    # plt.show()
    return gsPr_list, Z[neuron_num] #Z need to return z for plot_tuningcurves_sampled below
#peak_idx = np.argmax(Z_dog)        
                # peak_coords = np.unravel_index(peak_idx, Z_dog.shape)  # gets (i, j)
                # peak_value = Z_dog[peak_coords]
                #needs an array for the y_true parameter
                #should i be testing the location of the peak or the value of the peak?
                #location (index) i think
                # mse = mean_squared_error(Z[n][np.argmax(Z)],  Z_dog.flat[peak_idx])


                #  block_height = 2
                # block_width = 5
                # i_start = np.random.randint(0, len(config.exs[dim1]) - block_height + 1)
                # j_start = np.random.randint(0, len(config.exs[dim2]) - block_width + 1)
                # for di in range(block_height):
                #     for dj in range(block_width):
                #         i = i_start + di
                #         j = j_start + dj
def plot_tuningcurves_sampled_Penny(neuron_num, config):
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
    gsPr_list, Z = sampling_for_plots_Penny(neuron_num, config)
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
                # if f_peak is not None:
                #     plt.plot(f_peak[dim1], f_peak[dim2], 'yo', 
                #             label = f"GP offline peak: ({f_peak[dim1]:.2f}, {f_peak[dim2]:.2f})")
                plt.legend()
                plt.xlabel(f'Dimension {dim2 + 1}')
                plt.ylabel(f'Dimension {dim1 + 1}')
                plt.xticks(y_range)
                plt.yticks(x_range)
                plt.colorbar(label='Sampled Response')
                plt.title(f'Neuron {neuron_num} - Tuning Curves for Dimension {dim1 + 1} vs. Dimension {dim2 + 1}')
                plt.show()


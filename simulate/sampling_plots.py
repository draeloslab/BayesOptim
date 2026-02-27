import matplotlib.pylab as plt
import scipy.stats as stats
import numpy as np
from simulate.ground_truth_plot import plot_tuningcurves_Penny
from sklearn.metrics import mean_squared_error

# class sampling_plots():
#     def __init__(self):
#         self.means= means
#         self.covs=covs
def sampling_for_plots_Penny(config): 
    ''' 
    Grid Sample neural responses for the entire sample space

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
    d = config.d 
    SimPop = config.SimPop
    #Z_dog = plot_tuningcurves_Penny(N,SimPop,config) 
    #print(f"config_exs: {config.exs}")
    #max_tests = config.params['General']['max_tests']

    # for dim1 in range(len(config.exs)):
    #     for dim2 in range(dim1+1, len(config.exs)):
    #         X, Y= np.meshgrid(config.exs[dim1], config.exs[dim2], indexing = 'ij') 
    #         pos = np.dstack((X, Y))  
    #         Z = np.zeros((N, len(config.exs[dim1]) * len(config.exs[dim2])))
    #         #print(f"Z shape: {Z.shape}")

    x_range = np.arange(0, 10, 0.5)
    y_range = np.arange(0, 10, 0.5)
    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
    pos = np.dstack((X, Y))  # shape: (len(x_range), len(y_range), 2)

    num_x = len(x_range)
    num_y = len(y_range)

    #all_indices = [(i, j) for i in range(num_x) for j in range(num_y)]
    sample_indices = [(i, j) for i in range(1, num_x, 2) for j in range(1, num_y, 2)]  # start at 1, step by 2
    gsPr_list = []
    correct_neurons = 0
    #tests_so_far = 0

    #should I put a random seed here?
    for n in range(N):
        flag = False
        myflag = False
        sampled_responses = []
        # selected_indices = np.random.choice(len(all_indices), size=50, replace=False)
        # sample_indices = [all_indices[idx] for idx in selected_indices] 
        
        # num_points = len(config.exs[dim1]) * len(config.exs[dim2])
        # half_points = num_points //4
        # row_indices = np.arange(len(config.exs[dim1]))
        # random_rows = np.random.choice(row_indices, size=len(config.exs[dim1])//2, replace=False)
        #print(f"random rows {random_rows}")
        # indices_struct = [(i, j) for i in random_rows for j in range(len(config.exs[dim2]))]
        # np.random.shuffle(indices_struct)
        # for (i, j) in indices_struct[:half_points]:
        
        #print(f"indices_struct: {indices_struct}")
        #all_indices = [(i, j) for i in range(len(config.exs[dim1])) for j in range(len(config.exs[dim2]))]
        # for i in range(len(config.exs[dim1])): 
        #     for j in range(len(config.exs[dim2])): 
        #         resp = SimPop.sample(pos[i, j])[n]
        #         Z[n][i*len(config.exs[dim2])+ j] = resp
        # print(f"shape Z: {Z.shape}")
        #for (i, j) in all_indices:
        for (i, j) in sample_indices:
            gsPr_list.append(correct_neurons /N)
            peak_coords_pred = (i,j)
            #print(f"neuron {n} peak_coords_pred: {[peak_coords_pred]}")
            #peak_coords_true = np.unravel_index(np.argmax(Z_dog), Z_dog.shape)
            peak_coords_true = SimPop.peaks[n]
            mse = mean_squared_error(np.array(peak_coords_pred), np.array(peak_coords_true))
            dists = np.abs(np.array(peak_coords_pred), np.array(peak_coords_true))           
            count = np.count_nonzero(dists < SimPop.tol)  

            #if count > (d-1) and not flag:
                #flag = True

            #if mse < .05 and flag and not myflag: #8e-11:  #0.2
            # if mse <= config.mse_cutoff and flag and not myflag: #8e-11:  #0.2
            #     myflag = True
            #     correct_neurons+=1
            #     break
            if mse <= config.mse_cutoff:
            #     myflag = True
                correct_neurons+=1
                break
            # Append running probability 
    gsPr_list.append((correct_neurons) /N)
    #print(f"gsPr_list: {gsPr_list}")
    return gsPr_list #Z
    
    #here is where I sample the whole stimulus space
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
    #  
def plot_tuningcurves_sampled_Penny(config):
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
    gsPr_list, Z = sampling_for_plots_Penny(config)
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
                            origin='lower', cmap='viridis', aspect='auto')
                plt.plot(config.SimPop.peaks[n][0], config.SimPop.peaks[n][1], 'ro',
                        label = f"true peak: ({config.SimPop.peaks[n][0]:.2f}, {config.SimPop.peaks[n][1]:.2f})")
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
                plt.title(f'Neuron {n} - Tuning Curves for Dimension {dim1 + 1} vs. Dimension {dim2 + 1}')
                plt.tight_layout()
                plt.show()


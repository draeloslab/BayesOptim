import itertools
import math
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
import scipy.stats as stats
import numpy as np
from numpy import unravel_index

class Plot():
    def __init__(self,means,covs):
        self.means=means
        self.covs=covs

    def plot_prior(self,PCA_components,sample_size,nd,mean,cov,i=0): 
        '''
        Plots PCA-projection of high-dimensional multivariate normal distribution

        params:
        ---------
        PCA_components : PCA components
        sample_size    : points sampling from multivariate distirbution
        nd             : number of dimensions of multivariate normal distribution
        cov            : covariance of the distribution
        i (int)        : index of neuron (optional) 

        returns:
        ---------
        None

        ''' 
        if PCA_components <= nd:
            rv = np.random.multivariate_normal(mean, cov, sample_size)

            # Perform PCA to reduce the dimensionality to 2D for visualization
            pca = PCA(n_components=PCA_components)
            reduced_samples = pca.fit_transform(rv)

            # Create a scatter plot of the reduced samples
            plt.figure(figsize=(8, 6))
            plt.scatter(reduced_samples[:, 0], reduced_samples[:, 1], alpha=0.5)
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title(f'Neuron {i} PCA Projection of {nd}-dimensional Multivariate Normal Distribution')
            plt.grid(True)
            plt.show()
        else:
            print("PCA components are over dimensions of the prior distribution!")

    def plot_correct_prediction(N, Pr_list, rsPr_list, optimizer_name):
        ''' 
        Plots probability of making correct predictions

        params:
        ---------
        N (int)         : number of neurons 
        Pr_list (list)  : probability of correct predictions using Bayes Opt sampling
        rsPr_list (list): probability of correct predictions using random sampling

        returns:
        ---------
        None 

        '''

        plt.plot(np.arange(0,len(Pr_list)), Pr_list, linestyle='-', color='b',label="Bayes Opt")
        plt.plot(np.arange(0,len(rsPr_list)), rsPr_list, linestyle='-', color='c',label="Random Sampling")
        plt.xlabel('# of predictions')
        plt.ylabel('Probability')
        plt.title(f'Probability of making correct predictions - {optimizer_name}')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr = f" # of neurons = {N}"
        plt.text(0.50, 0.90, textstr,  fontsize=14, verticalalignment='top', bbox=props)
        
        plt.legend()
        plt.show()

    def plot_peak_value(x1, x2, mse_final, loc_list, SimPop, optimizer_name):
        ''' 
        Plots true peak value and predicted peak value

        params:
        ---------
        x1               : range of dimension 1
        x2               : range of dimension 2
        mse_final (list) : MSE values for each neuron
        loc_list (list)  : predicted peak locations
        SimPop (obj)     : simulated neurons class object

        returns:
        ---------
        None 

        '''

        positions = [index for index, value in enumerate(mse_final) if value > 1]

        #Predicted peak loc
        x1_coords = [point[0] for point in loc_list]
        x2_coords = [point[1] for point in loc_list]
        #True peak loc
        w1_coords = [point[0] for point in SimPop.peaks]
        w2_coords = [point[1] for point in SimPop.peaks]
        # Plot the points
        plt.scatter(x1_coords, x2_coords,label="Predicted")
        plt.scatter(w1_coords, w2_coords,label="True")
        plt.xticks(x1)
        plt.yticks(x2)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(f'Peak Value Loc in (x1,x2) {optimizer_name}')
        plt.legend()
        #plt.text(7, 13, 'MSE' , fontsize=22, bbox=dict(facecolor='red', alpha=0.5))
        plt.grid(True)

        # Label the specific points whose MSE is away from 0
        for index in positions:
            plt.text(x1_coords[index], x2_coords[index], str(index), fontsize=10, color='black', ha='center', va='center')
            plt.text(w1_coords[index], w2_coords[index], str(index), fontsize=10, color='red', ha='center', va='center')
        plt.show()

    def plot_mse(mse_final, optimizer_name):
        ''' 
        Plots MSE values of n neurons

        params:
        ---------
        mse_final (list) : MSE values for each neuron
        optimizer_name (string): name for the optimizer used

        returns:
        ---------
        None 

        '''
        x=[i for i in range(len(mse_final))]

        plt.plot(x, mse_final, marker='o', linestyle='', color='blue')
        plt.vlines(x, ymin=0, ymax=mse_final, color='blue', alpha=0.5)

        plt.xlabel('n-th Neurons')
        plt.ylabel('MSE')
        plt.title(f'MSE Plot - {optimizer_name}')
        plt.grid(True)
        plt.show()


    def plot_tuningcurves(N, exs, SimPop):
        ''' 
        Plot tuning curves as 2D contour plots for each dimension pair

        params:
        ---------
        N (int): number of neurons 
        exs    : dimension ranges
        SimPop : simulated neuron class object

        returns:
        ---------
        None 

        '''

        for dim1 in range(len(exs)):
            for dim2 in range(dim1+1, len(exs)):
                plt.figure(figsize=(8, 6))
                
                for n in range(N):
                    mean = SimPop.peaks[n][[dim1, dim2]]
                    cov = SimPop.covs[n][[dim1, dim2]][:, [dim1, dim2]]

                    x_range = exs[dim1]
                    y_range = exs[dim2]
                    X, Y = np.meshgrid(x_range, y_range, indexing = 'ij')

                    pos = np.dstack((X, Y))
                    Z = stats.multivariate_normal(mean=mean, cov=cov).pdf(pos)

                    plt.contourf(X, Y, Z, cmap='viridis', levels=20)

                plt.xlabel(f'Dimension {dim1 + 1}')
                plt.ylabel(f'Dimension {dim2 + 1}')
                plt.title(f'Tuning Curves for Dimension {dim1 + 1} vs. Dimension {dim2 + 1}')
                plt.colorbar()
                plt.show()

    def plot_tuningcurves_eval(N, exs, SimPop, candidates, method):
        ''' 
        Plot tuning curves as 2D contour plots for each dimension pair with predicted peak location plotted

        params:
        ---------
        N (int)           : number of neurons 
        exs               : dimension ranges
        SimPop            : simulated neuron class object
        candidates (list) : predicted peak locations
        method (list_)    : method of plotting candidate peak locations: "simulate", "manual", or "botorch"

        returns:
        ---------
        None 

        '''
        
        for n, neuron_candidates in enumerate(candidates):
            dim1 = 0
            dim2 = 1
            dim3 = 2
            # for dim1 in range(len(exs)):
            #     for dim2 in range(dim1 + 1, len(exs)):
            plt.figure(figsize=(8, 6))

            if method == 'simulate':
                plt.subplot(1,3,1)
                plt.imshow(SimPop.y[n][:,:,0])
                plt.scatter(neuron_candidates[dim2], neuron_candidates[dim1],color='red', marker='o', s=3)
                plt.ylabel('dim1')
                plt.xlabel('dim2')

                plt.subplot(1,3,2)
                plt.imshow(SimPop.y[n][:,0,:])
                plt.scatter(neuron_candidates[dim3], neuron_candidates[dim1],color='red', marker='o', s=3)
                plt.ylabel('dim1')
                plt.xlabel('dim3')

                plt.subplot(1,3,3)
                plt.imshow(SimPop.y[n][0,:,:])
                plt.scatter(neuron_candidates[dim3], neuron_candidates[dim2],color='red', marker='o', s=3)
                plt.ylabel('dim2')
                plt.xlabel('dim3')

                plt.show()

            else:
                mean = SimPop.peaks[n][[dim1, dim2]]
                cov = SimPop.covs[n][[dim1, dim2]][:, [dim1, dim2]]

                x_range = exs[dim1]
                y_range = exs[dim2]
                X, Y = np.meshgrid(x_range, y_range, indexing = 'ij')

                pos = np.dstack((X, Y))
                Z = stats.multivariate_normal(mean=mean, cov=cov).pdf(pos)

                plt.contourf(X, Y, Z, cmap='viridis', levels=20)

                # Overlay candidate points on the contour plot for the current neuron (n)
                if method == 'botorch':
                    candidate_x = [candidate[dim1] for candidate in neuron_candidates]
                    candidate_y = [candidate[dim2] for candidate in neuron_candidates]
                elif method == 'manual':
                    candidate_x = candidates[n][dim1]
                    candidate_y = candidates[n][dim2]
                plt.scatter(candidate_x, candidate_y, color='red', marker='o', s=3)  # 'o' for dot

                plt.xlabel(f'Dimension {dim1 + 1}')
                plt.ylabel(f'Dimension {dim2 + 1}')
                plt.title(f'Neuron {n+1} - Tuning Curves for Dimension {dim1 + 1} vs. Dimension {dim2 + 1}')
                plt.colorbar()
                plt.show()

    def plot_tuningcurves_pseudo(pred_means, exs, dataset_type, neuron, eval=False):
        ''' 
        Plot tuning curves as 2D contour plots for each dimension pair with predicted peak location plotted

        params:
        ---------
        pred_means (list) : predicted means for each stimulus type 
        exs               : dimension ranges
        dataset_type (str): "offline", "dataset", "population average"
        neuron (int)      : neuron number
        eval (bool)       : if True, will plot the predicted peak location (optional)

        returns:
        ---------
        None 

        '''
        stim_params = ['Funkiness', 'Orientation', 'Contrast']
        dim_pairs = [(0,1), (1,2), (0,2)]
        fig_raw, axs_raw = plt.subplots(1,3, figsize=(20, 5), facecolor='w', edgecolor='k')
        fig_interp, axs_interp = plt.subplots(1,3, figsize=(20, 5), facecolor='w', edgecolor='k')

        for i, (dim1, dim2) in enumerate(dim_pairs):
            x1, x2 = np.meshgrid(exs[dim1], exs[dim2])
            extents = (x1.min(), x1.max(), x2.min(), x2.max())
            peaks = np.unravel_index(np.argmax(pred_means[i]), pred_means[i].shape)

            contour_raw = axs_raw[i].imshow(pred_means[i].T, cmap='viridis', origin='lower', extent=extents)
            fig_raw.colorbar(contour_raw, ax=axs_raw[i])
            axs_raw[i].set_xlabel(f'{stim_params[dim1]}')
            axs_raw[i].set_ylabel(f'{stim_params[dim2]}')
            axs_raw[i].set_title(f'{stim_params[dim1]} vs {stim_params[dim2]}')

            contour_interp = axs_interp[i].imshow(pred_means[i].T, cmap='viridis', origin='lower', extent=extents, interpolation='bilinear')
            fig_interp.colorbar(contour_interp, ax=axs_interp[i])
            axs_interp[i].set_xlabel(f'{stim_params[dim1]}')
            axs_interp[i].set_ylabel(f'{stim_params[dim2]}')
            axs_interp[i].set_title(f'{stim_params[dim1]} vs {stim_params[dim2]}')

            if eval:
                axs_interp[i].scatter(peaks[0], peaks[1], c='red', s=100)

        fig_raw.suptitle(f'Raw Contour plots for {neuron} ({dataset_type})')
        fig_interp.suptitle(f'Bilinear Interpolation Contour plots for {neuron} ({dataset_type})')
        plt.tight_layout()
        plt.show()

    def plot_tuningcurves_sampled(neuron_num, config, f_peak = None):
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
        Z = sampling_for_plots(neuron_num, config)
        sampled_peak = config.x_star[np.argmax(Z)] #unravel_index(Z.argmax(), np.transpose(Z.shape))
        for dim1 in range(len(config.exs)):
            for dim2 in range(dim1+1, len(config.exs)):
                for n in range(neuron_num, neuron_num+1):
                    plt.figure(figsize=(8, 6))
                    x_range = config.exs[dim1]
                    y_range = config.exs[dim2]
                    X, Y = np.meshgrid(x_range, y_range, indexing = 'ij')
                    Z_reshaped = Z.reshape(X.shape).T
                    plt.imshow(Z_reshaped, extent=(x_range[0] - 0.5, x_range[-1] + 0.5, y_range[0]-0.5, y_range[-1]+0.5),
                                origin='lower', cmap='viridis', aspect='auto')
                    plt.plot(config.SimPop.peaks[neuron_num][0], config.SimPop.peaks[neuron_num][1], 'ro',
                            label = f"true peak: ({config.SimPop.peaks[neuron_num][0]:.2f}, {config.SimPop.peaks[neuron_num][1]:.2f})")
                    plt.plot(sampled_peak[dim1], sampled_peak[dim2], 'bo',
                            label = f"sampled peak: ({sampled_peak[dim1]:.2f}, {sampled_peak[dim2]:.2f})")
                    if f_peak is not None:
                        plt.plot(f_peak[dim1], f_peak[dim2], 'yo', 
                                label = f"GP offline peak: ({f_peak[dim1]:.2f}, {f_peak[dim2]:.2f})")
                    plt.legend()
                    plt.xlabel(f'Dimension {dim1 + 1}')
                    plt.ylabel(f'Dimension {dim2 + 1}')
                    plt.xticks(x_range)
                    plt.yticks(y_range)
                    plt.colorbar(label='Sampled Response')
                    plt.title(f'Neuron {neuron_num} - Tuning Curves for Dimension {dim1 + 1} vs. Dimension {dim2 + 1}')
                    plt.show()

    def sampling_for_plots(neuron_num, config):
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
                for i in range(len(config.exs[dim1])):  # 16
                    for j in range(len(config.exs[dim2])):  # 10
                        # putting the set seed here kind of lose the sampling power??
                        # np.random.seed(config.params['General']['seed'])
                        # print(pos[i,j])
                        resp = config.SimPop.sample(pos[i,j])[neuron_num]
                        # print(resp)
                        Z[i*len(config.exs[dim2])+ j] = resp # just sampling
                # # sample for nD array; discarded
                # Z = np.zeros((len(exs[dim2]), len(exs[dim1])))
                # for i in range(Z.shape[dim1]):  # 10
                #     for j in range(Z.shape[dim2]):  # 16
                #         Z[i,j] = SimPop.sample(pos[i,j])[neuron_num]  # sample from the population
        return Z

    # this plots for posterior mean f
    def plot_posterior_mean(neuron_num, exs, f_all, double_peaks = False): 
        ''' 
        Plot posterior mean for a specific neuron

        params:
        ---------
        neuron_num (int): index of a specific neuron
        exs    : dimension ranges
        f_all  : posterior mean for all neurons
        double_peaks : if want to detect more than one peak (optional)

        returns:
        ---------
        None

        '''
        for dim1 in range(len(exs)):
            for dim2 in range(dim1+1, len(exs)):
                X, Y= np.meshgrid(exs[dim1], exs[dim2], indexing = 'ij') 
                pos = np.dstack((X, Y))
                if len(f_all[neuron_num]) > 20: 
                    print("Sample length greater than 20, only print the last 20 posterior means")
                    plot_range = slice(-20, None)
                else:
                    plot_range = slice(0, len(f_all[neuron_num]))
                for Z in f_all[neuron_num][plot_range]:
                    # TODO: fix the config situation
                    if double_peaks:
                        top_two_peaks = detect_two_peaks(Z)  # TODO: add detect_two_peaks here
                        pl_ls = np.array([config.x_star[pl] for pl in top_two_peaks])
                    Z_reshaped = Z.reshape(X.shape).T
                    plt.figure(figsize=(8, 6))
                    plt.imshow(Z_reshaped, #extent=(x_range[0], x_range[-1], y_range[0], y_range[-1]),
                                origin='lower', cmap='viridis', aspect='auto')
                    if double_peaks:
                        for value in pl_ls:
                            plt.plot(value[0], value[1], 'ro', label=f"Peak: ({value[0]:.2f}, {value[1]:.2f})")
                    # contour = plt.contourf(X, Y, Z_mine, levels=20, colors = 'white')
                        plt.legend()
                    plt.xlabel(f'Dimension {dim1 + 1}')
                    plt.ylabel(f'Dimension {dim2 + 1}')
                    plt.colorbar(label='Sampled Response')
                    plt.title('Posterior Mean - Neuron ' + str(neuron_num))
                    plt.show()

    def plot_stopping_criteria(stopping_allN, stopping_crit, EI_or_PI):
        ''' 
        Plots Expected Improvement optimization value (stopping value) across number of tests for all neurons

        params:
        ---------
        stopping_allN (list) : stopping values (EI and PI optimization value) for all tests for all neurons
        stopping_crit : the stopping criterion value set by config files
        EI_or_PI: indicate "EI" or "PI" to get specific stopping criteria

        returns:
        ---------
        None 

        '''
        stop_idx = 0 if EI_or_PI == "EI" else 1  # EI idx = 0; PI idx = 1; 
        N = len(stopping_allN)  # number of neurons

        # decide how many rows/cols to make the subplot grid
        rows = int(math.ceil(math.sqrt(N)))
        cols = int(math.ceil(N / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(rows*2, cols*2))
        if N == 1:
            axes = [axes]
        else:
            # flatten so we can index it with a single [i]
            axes = axes.flatten()

        for i in range(N):  
            ax = axes[i] 
            noise_EI = []
            if len(stopping_allN[i]) < 2:
                ax.plot(stopping_allN[i][0][stop_idx], 'bo')
            else:
                for j in range(len(stopping_allN[i])):
                    # Condition for coloring
                    # color = 'blue' if correct_sol_plot[i][j]==True else 'red'  # blue: correct solution
                    # ax.plot(j, file_to_plot[i][j][EI_idx], 'o', color=color)
                    noise_EI.append(stopping_allN[i][j][stop_idx])
                ax.plot(noise_EI, alpha = 0.5)        # Plot the EI trace for neuron i
                # ax2 = ax.twinx()
                # ax2.plot(dist_to_plot[i], alpha=0.5, linestyle="-.")  # distance to true peak
            ax.axhline(y = stopping_crit,color='black', linestyle='--')  # the stopping criteria
        for k in range(N, len(axes)):
            axes[k].set_visible(False)
        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()

    # def plot_just_stopping(stopping_list):
    #     ''' 
    #     Plots Expected Improvement optimization value (stopping value) across number of tests for each neuron

    #     params:
    #     ---------
    #     stopping_list (list) : stopping value (EI optimization value) for each test for each neuron

    #     returns:
    #     ---------
    #     None 

    #     '''
    #     plt.figure(figsize=(8, 6))
    #     for n in range(len(stopping_list)):
    #         if len(stopping_list[n])>1:
    #             plt.plot(stopping_list[n])
    #     plt.title(f'{len(stopping_list)} neurons - Stopping Criteria')
    #     plt.xlabel('Number of Tests')
    #     plt.ylabel('Stopping criteria')
    #     plt.show()

    def plot_acqf(acq_list):
        ''' 
        Plots values for the Upper Confidence Bound acquisition function for each neuron for each test

        params:
        ---------
        acq_list (list) : acquisition values from the UCB function for each neuron

        returns:
        ---------
        None 

        '''

        acqf_list = [[float(acq) for acq in acq_sublist] for acq_sublist in acq_list]
        for n in range(len(acqf_list)):
            plt.figure(figsize=(8, 6))
            plt.plot(acqf_list[n])
            plt.title(f'Neuron {n} - UCB Acquisition Function')
            plt.xlabel('Number of Tests')
            plt.ylabel('AcqF criteria')

    def plot_mse_runtime_map(MSE,runt_list,vmin,vmax):
        ''' 
        Plots run time for each neuron with a colorbar representing MSE values

        params:
        ---------
        MSE (list)       : MSE values for each neuron
        runt_list (list) : list of run time values for each neuron
        vmin (tuple)     : minimum MSE value
        vmax (tuple)     : maximum MSE value

        returns:
        ---------
        None 

        '''
        # Create a colormap (blue to red)
        cmap = plt.get_cmap('coolwarm')

        # Normalize the MSE values to be in the range [0, 1]
        norm = plt.Normalize(vmin, vmax)

        fig, ax = plt.subplots()
        bars = ax.bar(range(len(runt_list)), runt_list, color=cmap(norm(MSE)))
        plt.xlabel('n-th Neurons')
        plt.ylabel('Runtime')
        plt.title('Runtime Plot with MSE' )

        # Add a colorbar to indicate the MSE values
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax)
        cbar.set_label('MSE')
        # ax.set_ylim(0, ymax) 
        plt.show()

    def plot_run_time(test_time_neuron, neuron=None, average=False):
        ''' 
        Plots run time for test for a specific neuron

        params:
        ---------
        test_time_neuron (list) : list of run time value for each test; can be a list of lists for each neuron if len(neuron) > 1
        neuron (list)           : list neuron indices
        average (bool)          : if True, averages test time for each test across neurons
        
        returns:
        ---------
        None 

        '''

        if average:
            max_length = max(len(lst) for lst in test_time_neuron)
            averages = []
            for i in range(max_length):
                sum_idx = 0
                count_idx = 0
                for test_time in test_time_neuron:
                    if i < len(test_time):
                        sum_idx += test_time[i]
                        count_idx += 1
                averages.append(sum_idx / count_idx)

            averages_scaled = [average * (0.035/max(averages)) for average in averages]

            plt.figure()
            plt.plot(range(len(averages_scaled)), averages_scaled, linewidth=3)
            plt.axhline(y=0.035, color='orange', linestyle='--')
            plt.ylim([0,0.05])
            # plt.xticks(range(0, len(averages), 5), range(0, len(averages), 5))
            plt.xlabel('# of Tests')
            plt.ylabel('Time (sec)')
            plt.title('Average time per test')
        
        else:
            length = len(neuron)
            fig, axs = plt.subplots(length,1, figsize=(10, 2*length), facecolor='w', edgecolor='k')
            fig.suptitle('Run time per test for each neuron')
            if length > 1:
                axs = axs.ravel()
                for i, n in enumerate(neuron):
                    axs[i].bar(range(len(test_time_neuron[n])), test_time_neuron[n])
                    axs[i].set_xticks(range(100), [str(i+1) for i in range(100)])
                    # axs[i].set_xticks(range(len(test_time_neuron[n])), [str(i+1) for i in range(len(test_time_neuron[n]))])
                    axs[i].set_xlabel('Number of tests')
                    axs[i].set_ylabel('Time (sec)')
                    axs[i].set_title(f'Neuron {n}')
            else:
                axs[i].bar(range(len(test_time_neuron[neuron[0]])), test_time_neuron[neuron[0]])
                axs[i].set_xticks(range(len(test_time_neuron[0])), [str(i+1) for i in range(len(test_time_neuron[0]))])
                axs[i].set_xlabel('Number of tests')
                axs[i].set_ylabel('Time (sec)')
                axs[i].set_title(f'Neuron {neuron[n]}')

        plt.tight_layout()
        plt.show()

        




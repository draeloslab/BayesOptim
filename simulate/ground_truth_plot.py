import matplotlib.pylab as plt
import scipy.stats as stats
import numpy as np

# class ground_truth_plot():
#     def __init__(self):
#         self.means= means
#         self.covs=covs

def plot_tuningcurves_Penny(N, SimPop, config):
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
    for dim1 in range(len(config.exs)):
        for dim2 in range(dim1+1, len(config.exs)):
            plt.figure(figsize=(8, 6))
            for n in range(N):
                if config.params['Neurons']['SimPop'] == "auditory": 
                    mean_exc = SimPop.mean1[n, [dim1, dim2]]     
                    cov_exc = SimPop.covs1[n][[dim1, dim2]][:, [dim1, dim2]] 
                    mean_inh = SimPop.mean2[n, [dim1, dim2]]      
                    cov_inh = SimPop.covs2[n][[dim1, dim2]][:, [dim1, dim2]]
                    x_range = config.exs[dim1]
                    y_range = config.exs[dim2]
                    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
                    pos = np.dstack((X, Y))

                    Z_exc = stats.multivariate_normal(mean=mean_exc, cov=cov_exc).pdf(pos)
                    Z_inh = stats.multivariate_normal(mean=mean_inh, cov=cov_inh).pdf(pos)
                    Z_dog = Z_exc - Z_inh

                    print(f"Z_dog neuron {n}: {Z_dog}; Z_dog shape: {Z_dog.shape}")

                    plt.figure()
                    plt.contourf(Z_dog, cmap='viridis', 
                               origin='lower') #,  
                            #    extent=[x_range[0] - 0.5, x_range[-1] + 0.5, y_range[0]-0.5, y_range[-1]+0.5])
                    plt.title(f'Ground Truth Neuron {n} DoG Response')
                    plt.colorbar()
                    plt.xticks(y_range)
                    plt.yticks(x_range)
                    plt.show()
                    
                else:
                    mean = SimPop.peaks[n][[dim1, dim2]]
                    cov = SimPop.covs[n][[dim1, dim2]][:, [dim1, dim2]]

                    x_range = config.exs[dim1]
                    y_range = config.exs[dim2]
                    X, Y = np.meshgrid(x_range, y_range, indexing = 'ij')

                    pos = np.dstack((X, Y))
                    Z = stats.multivariate_normal(mean=mean, cov=cov).pdf(pos)

                    plt.contourf(X, Y, Z, cmap='viridis', levels=20)
                    plt.xlabel(f'Dimension {dim1 + 1}')
                    plt.ylabel(f'Dimension {dim2 + 1}')
                    plt.title(f'Tuning Curves for Dimension {dim1 + 1} vs. Dimension {dim2 + 1}')
                    plt.colorbar()
                    plt.show()

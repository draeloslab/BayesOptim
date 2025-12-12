from simulate.neuron import Neuron

import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_squared_error

class SimNeurons(Neuron):
    ''' 
    Class to simulate neural tuning curves and responses to visual stimuli
    '''
    def __init__(self, N, d, tol=1):
        ''' 
        Initialize the SimNeurons instance

        params:
        ----------
        N (int)   : number of neurons
        d (int)   : number of dimensions
        tol (tuple) : distance tolerance for peak locations
        '''    
        super().__init__(N, d, tol)
        self.y = []

    def gen_tuning_curves(self, type='indep', constraint=None):
        ''' 
        Generates tuning curves for N number of neurons from a multivariate normal distribution with a random mean and covariance

        params:
        ----------
        type (str): "indep", "corr", "simulate"
        constraint (str): If not None, "linear"

        '''
        if not constraint: constraint = ''
        self.type = type

        # currently assuming all single peaked tuning curves  <- wait who made this comment
        means = np.zeros((self.N,self.d))
        covs = np.zeros((self.N,self.d,self.d))

        if type == 'indep' or type == 'indep_noise':  # currently assume noise are just to each neuron
            # means = np.zeros((self.N,self.d))
            # covs = np.zeros((self.N,self.d,self.d))
            for i in range(self.d):
                means[:,i] = self.min[i] + np.random.random(size=self.N) * self.scale[i] # min + random increment = mean
                #random.random(N) array with len=N from (0,1)
            
            if constraint == "linear": # add linear constraint y: Last element of mean=Y < Σ (xi) + 0.75*scale & Y > Σ (xi) - 0.75*scale
                for n in range(self.N): 
                    while not ((sum(means[n][:-1])+0.75*self.scale[-1] > means[n][-1]) and (sum(means[n][:-1])-0.75*self.scale[-1] < means[n][-1])):
                            for i in range(self.d):
                                means[n,i] = self.min[i] + np.random.random() * self.scale[i]
                    #print("mean",means[n])

            for i in range(self.d):
                # covs[:,i,i] = (self.count[i]*1e8) * self.scale[i]**2 #np.random.random(size=self.N) * self.scale[i]
                covs[:,i,i] = np.random.random(size=self.N) * self.scale[i]**2 / np.sqrt(self.count[i]) # diagonal elements of the covariance matrices

            for n in range(self.N):
                self.y.append(stats.multivariate_normal(mean=means[n], cov=covs[n]))
                #print("means[n] is", means[n], "covs[n] is", covs[n] )
            self.peaks = means 
            self.covs=covs
          
        elif type == 'corr':
            pass
        
        if type == 'simulate':
            # This is the case where we want to generate a simulated dataset that is similar to what we'd expect from a RBF x linear kernel combination
            # Dimension sizes are (N, d-1) since we will add on the 3rd dimension with linear scaling later
            means = np.zeros((self.N, self.d-1))
            covs = np.zeros((self.N, self.d-1, self.d-1))
            for i in range(self.d-1):
                means[:,i] = self.min[i] + np.random.random(size=self.N) * self.scale[i]
            
            if constraint == 'linear':
                for n in range(self.N): 
                    while not ((sum(means[n][:-1])+0.75*self.scale[-1] > means[n][-1]) and (sum(means[n][:-1])-0.75*self.scale[-1] < means[n][-1])):
                            for i in range(self.d-1):
                                means[n,i] = self.min[i] + np.random.random() * self.scale[i]
            
            for i in range(self.d-1):
                covs[:,i,i] = np.random.random(size=self.N) * (self.scale[i])**2 / np.sqrt(self.count[i])
            
            multi_dist = []
            for n in range(self.N):
                multi_dist.append(stats.multivariate_normal(mean=means[n], cov=covs[n]))

            peaks = []
            
            # Generating a "3d cube" - dimensions 1 and 2 are periodic, and dimension 3 is the linear component
            # Creates a meshgrid of the dim1-dim2 parameter space and then generates samples by taking the PDF
            # Linearly scales and stacks them to add the linear component
            self.y = np.zeros((self.N, self.count[0], self.count[1], self.count[2]))   
            for n in range(self.N): 
                z = np.zeros((int(self.count[0]), int(self.count[1])))
                pos = np.dstack(np.meshgrid(np.arange(self.count[0]), np.arange(self.count[1]), indexing='ij')) # meshgrid of dim1-dim2 parameter space

                z = 20 * multi_dist[n].pdf(pos) # generating samples 
                z += np.random.random(1)*0.1/np.max(self.scale)
                
                # Adding the "linear kernel component" - scaling values and creating a 3d cube
                zrv = np.random.randint(-10,10) / 10
                slope_values = np.full_like(z, 1)
                factors = zrv*np.arange(-0.25,0.25)
                modified_samples = np.zeros((self.count[0], self.count[1], factors.shape[0])) 
                for i, factor in enumerate(factors):
                    modified_samples[:,:,i] = z + slope_values*factor

                self.y[n] = modified_samples  
                peaks.append(np.unravel_index(np.argmax(self.y[n]), self.y[n].shape)) # peaks is the argmax location for each neuron

            ## Adds noise to make slightly more biorealisitc 
            # noise = np.random.normal(scale=0.001, size=self.y.shape)
            # self.y += noise

            self.peaks = np.array(peaks)
            self.covs = covs

        elif type == 'linear_uvn':
            for i in range(self.d):
                means[:,i] = self.min[i] + np.random.random(size=self.N) * self.scale[i] # min + random increment = mean
                #random.random(N) array with len=N from (0,1)
                # covariance is huge
                covs[:,i,i] = np.random.random(size=self.N) * self.scale[i]**2 / np.sqrt(self.count[i])

            if constraint == "linear": # add linear constraint y: Last element of mean=Y < Σ (xi) + 0.75*scale & Y > Σ (xi) - 0.75*scale
                for n in range(self.N): 
                    while not ((sum(means[n][:-1])+0.75*self.scale[-1] > means[n][-1]) and (sum(means[n][:-1])-0.75*self.scale[-1] < means[n][-1])):
                            for i in range(self.d):
                                means[n,i] = self.min[i] + np.random.random() * self.scale[i]
                    #print("mean",means[n])
            # generate data, 1d linear
            slopes = np.random.uniform(0.1, 1.1, size=self.N)  # may have to change the slope range, no negative numbers are allowed
            intercepts = np.random.uniform(self.min[0], self.max[0], size=self.N)  # first dim
            peaks = np.zeros((self.N, self.d))
            peaks[:,0] =  self.max[0] - np.random.random(size=self.N) * self.scale[0]/10 #slopes[n] * self.max[0] +intercepts[n]  # for first dim
            peaks[:,1:] = means[:, 1:]
            for n in range(self.N):
            #     if slopes[n] < 0 :
            #         peaks[n,0] = self.min[0] + np.random.random(1) * self.scale[0]/10
            #     else: # slopes[n] >= 0
            #         peaks[n,0] = self.max[0] - np.random.random(1) * self.scale[0]/10
                y_linear = (slopes[n], intercepts[n])  # first dim
                y_uvn = [stats.norm(loc=means[n, i], scale=np.sqrt(covs[n, i, i])) for i in range(1, self.d)]  # other dim
                self.y.append((y_linear, y_uvn))
        
            self.peaks = peaks   # I don't think this would work
            self.covs=covs
        elif type == "double_peaks":  # neurons with 2 peaks (maximum number of 2 mvn distributions)
            means_2 = np.zeros((self.N,self.d))
            for i in range(self.d):
                means[:,i] = self.min[i] + np.random.random(size=self.N) * self.scale[i] # min + random increment = mean
                means_2[:,i] = self.max[i] - np.random.random(size=self.N) * self.scale[i] #self.max[i] - np.random.random(size=self.N) * self.scale[i]
                #random.random(N) array with len=N from (0,1)
            
            if constraint == "linear": # add linear constraint y: Last element of mean=Y < Σ (xi) + 0.75*scale & Y > Σ (xi) - 0.75*scale
                for n in range(self.N): 
                    while not ((sum(means[n][:-1])+0.75*self.scale[-1] > means[n][-1]) and (sum(means[n][:-1])-0.75*self.scale[-1] < means[n][-1])):
                        for i in range(self.d):
                            means[n,i] = self.min[i] + np.random.random() * self.scale[i]
                    #print("mean",means[n])
            elif constraint == "peak_distance":
                # raise ValueError("an error!")
                for n in range(self.N):
                    while not (np.linalg.norm(means[n]-means_2[n]) >= 3):
                        for i in range(self.d):
                            means_2[n,i] = self.min[i] + np.random.random() * self.scale[i] #self.max[i] - np.random.random() * self.scale[i]

            for i in range(self.d):
                # covs[:,i,i] = (self.count[i]*1e8) * self.scale[i]**2 #np.random.random(size=self.N) * self.scale[i]
                covs[:,i,i] = np.random.random(size=self.N) * self.scale[i]**2 / np.sqrt(self.count[i]) # diagonal elements of the covariance matrices
            for n in range(self.N):
                peak1 = stats.multivariate_normal(mean=means[n], cov=covs[n])
                peak2 = stats.multivariate_normal(mean=means_2[n], cov=covs[n])
                self.y.append((peak1, peak2))
                #print("means[n] is", means[n], "covs[n] is", covs[n] )
            self.peaks = (means, means_2) 
            self.covs= covs 
            # print(means, means_2, flush=True)

    # sampling is across all neurons, each neuron will have one response value when inputted
    # (d,1) x sample point
    def sample(self, x_sample, normalize=False):
        ''' 
        Generates a simulated z response an x sample stimulus
        
        params:
        ----------
        x (tuple)        : sample stimulus
        normalize (bool) : if True, normalize response by the maximum response

        returns:
        ----------
        z                : simulated response given sample stimulus
        '''
    #-------note: given we can actually only sample of a small portion of the sample space.
    #-------Input: x sample point =>generate response z
        ### a local seed with local RNG
        seed = hash(tuple(x_sample)) % (2**32)
        local_rng = np.random.default_rng(seed)
        ###
        z = np.zeros(self.N)
        for i in range(self.N):#i,y in enumerate(self.y):
            if isinstance(self.y[i], tuple):  # check if it's indep or (linear_uvn or double_peaks)
                if all(hasattr(comp, 'pdf') for comp in self.y[i]):  # double_peaks; need to uncomment this
                    peak1, peak2 = self.y[i]
                    z_1 = peak1.pdf(x_sample)  # .pdf(pos)
                    z_2 = peak2.pdf(x_sample)
                    z[i] = 20 * np.maximum(z_1, z_2)
                else:  # linear_uvn
                    (a,b), y_uvn = self.y[i]
                    z_linear = (a * x_sample[0] + b)  #/20  # not sure what does the 20 means
                    z_uvn = np.prod([dist.pdf(x_sample[1:]) for dist in y_uvn])  # times all other dim
                    z[i] = z_linear * z_uvn  # do we need to *20 here?
            else:  # indep
                z[i] = 20 * self.y[i].pdf(x_sample) # need to uncomment this
            
            # adding noise 
            # TODO: should we distinguish bewteen sample noise and intrinsic noise
            if self.type == "indep_noise":
                # # z[i] += np.random.poisson(lambda = )  # poisson noise is quite disturbing
                alpha = 0.5  # change this
                noise_std = z[i] * alpha
                z[i] += local_rng.normal(0, noise_std)  # comment this if rng is not working
                # z[i] += np.random.normal(0, noise_std)
                if z[i] < 0:
                    z[i] = 0

            else:
                z[i] += local_rng.random()*0.1/np.max(self.scale) 
                #----noise--gaussian
                # z[i] = np.random.poisson(fr)
        
        # # to account for negative values, should be with respect to each neuron, not wrt every neurons
        # if np.min(z) < 0:
        #     z = z + abs(np.min(z))
            
        if normalize:
            self.record_response(x_sample, z, normalize=True)

        self.record_response(x_sample, z)
        return z
 
    
    def verify_sln_double(self, peaks, n):
        # self.peaks shape: (N, d)
        #-- (number of neurons, number of dimensions in tuning curve)
        # peaks shape: (2,2), it's a list becuase we have two peaks
        # this is for double peaks only
        peak1 = self.peaks[0][n]
        peak2 = self.peaks[1][n]
        dists_1 = np.abs(peaks - peak1)
        dists_2 = np.abs(peaks - peak2)
        filtered_dists_1 = dists_1[0] if np.sum(dists_1[0]) < np.sum(dists_1[1]) else dists_1[1]
        filtered_dists_2 = dists_2[0] if np.sum(dists_2[0]) < np.sum(dists_2[1]) else dists_2[1]

        dists_ls = [filtered_dists_1, filtered_dists_2]
        mse_ls = [mean_squared_error(filtered_dists_1, peak1), mean_squared_error(filtered_dists_2, peak2)]
        count_ls = [np.count_nonzero(filtered_dists_1 < self.tol), np.count_nonzero(filtered_dists_2 < self.tol)]
        # selected_peak = peak1 if np.sum(dists_1) < np.sum(dists_2) else peak2
        # dists = dists_1 if np.array_equal(selected_peak, peak1) else dists_2
        # mse = mean_squared_error(peaks, selected_peak)  # double-peaks only
        # count = np.count_nonzero(dists < self.tol) #--count dist within tolerance
        return dists_ls, count_ls, mse_ls #dists, count, mse #
    
    def get_prior(self):
        return  self.peaks, self.covs
    
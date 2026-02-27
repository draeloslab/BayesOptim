import numpy as np
import scipy.stats as stats
import pandas as pd
from sklearn.metrics import mean_squared_error
from simulate.neuron import Neuron
import matplotlib.pylab as plt

class Auditory_neurons(Neuron):
    ''' 
    Class to simulate neural tuning curves and responses to auditory stimuli
    '''
    def __init__(self, N, d, tol=1):
        ''' 
        Initialize the Auditory_neurons instance

        params:
        ----------
        N (int)   : number of neurons
        d (int)   : number of dimensions
        tol (tuple) : distance tolerance for peak locations
        '''    
        super().__init__(N, d, tol)
        #self.y = []
        
    def gen_tuning_curves(self, type='indep', constraint = None):
        ''' 
        Generates tuning curves for N number of neurons from a multivariate normal distribution with a random mean and covariance

        params:
        ----------
        type (str): "indep", "corr", "simulate"
        constraint (str): If not None, "linear"

        '''
        self.type = type
        self.rv1 = []  # List to store "excitation" random variable
        self.rv2 = []  # List to store "inhibition" random variable
        self.mean1 = np.zeros((self.N,self.d)) #smaller variance 
        self.covs1 = np.zeros((self.N,self.d,self.d))
        self.mean2 = np.zeros((self.N,self.d))  #larger variance
        self.covs2 = np.zeros((self.N,self.d,self.d))
    
        self.offset = 1
        center = np.array([(r - 1) / 2 for r in self.ranges])
        sets = []

        # Set 0: Center for excitation/inhibition
        mean_exc1 = center.copy()
        mean_inh1 = center.copy()
        cov_exc1 = np.eye(self.d) * (1.0 / self.d)
        cov_inh1 = np.eye(self.d) * (5.0 /self.d)

        # Set 1: Center + offset in dimension 0
        mean_exc2 = center.copy(); mean_exc2[0] += self.offset
        mean_inh2 = center.copy(); mean_inh2[0] += self.offset +2
        cov_exc2 = np.eye(self.d) * (1.0 / self.d)
        cov_inh2 = np.eye(self.d) * (3.0 /self.d)

        # Set 2: Center + offset in dimension 1
        mean_exc3 = center.copy()
        mean_inh3 = center.copy()
        if self.d > 1:
            mean_exc3[1] += self.offset
            mean_inh3[1] += self.offset
            cov_exc3 = np.eye(self.d) * (1.0 / self.d)
            cov_inh3 = np.eye(self.d) * (4.0 /self.d)

        sets.append((mean_exc1, mean_inh1, cov_exc1, cov_inh1))
        sets.append((mean_exc2, mean_inh2, cov_exc2, cov_inh2))
        sets.append((mean_exc3, mean_inh3, cov_exc3, cov_inh3))

        self.peak_types = []
        np.random.seed(40)
        #same code as config
        xs = np.meshgrid(*self.x, indexing='ij')
        x_star = np.empty(xs[0].shape + (self.d,))
        for i in range(self.d):
            x_star[..., i] = xs[i]
        self.x_star = x_star.reshape(-1, self.d)
        self.peaks = np.zeros((self.N, self.d))
        self.min = np.zeros((self.N, self.d))
        self.responses = []
        for n in range(self.N):
            idx = np.random.choice(3)
            mean_exc, mean_inh, cov_exc, cov_inh = sets[idx]
            self.mean1[n] = mean_exc
            self.covs1[n] = cov_exc
            self.mean2[n] = mean_inh  #I am not doing anything with this
            self.covs2[n] = cov_inh
            self.peak_types.append(idx)
            print(f"Neuron {n} is peak type {idx}")
            
            self.rv1.append(stats.multivariate_normal(mean=self.mean1[n], cov=self.covs1[n]))
            #self.rv2.append(stats.multivariate_normal(mean=self.mean2[n], cov=self.covs2[n]))
            sideband1 = stats.multivariate_normal(mean= self.mean1[n]- self.offset, cov=self.covs2[n])
            sideband2 = stats.multivariate_normal(mean= self.mean1[n] + self.offset, cov=self.covs2[n])
            self.rv2.append(lambda x, sb1=sideband1, sb2=sideband2: sb1.pdf(x) + sb2.pdf(x))
            self.responses.append([self.rv1[n].pdf(xi) - self.rv2[n](xi) for xi in self.x_star])
            # inhibition_grid = [sideband1.pdf(xi) + sideband2.pdf(xi) for xi in self.x_star]
            # self.rv2.append(np.array(inhibition_grid))
            # excitation_grid = [self.rv1[n].pdf(xi) for xi in self.x_star]
            # dog_response = np.array(excitation_grid) - self.rv2[n]
            # self.responses.append(dog_response.tolist())
            #self.responses.append([self.rv1[n].pdf(xi) - self.rv2[n]])
            peak_idx = np.argmax(self.responses[n])
            min_idx = np.argmin(self.responses[n])
            self.min[n] = self.x_star[min_idx]
            self.peaks[n] = self.x_star[peak_idx]
            print(f"self.peaks: neuron {n} {self.peaks[n]}")

            # print(f'Excitatory mean: {self.mean1[n]}')
            ##print(f'Inhibitory mean {self.mean2[n]}')
            # print(f'Excitatory covariance: {self.covs1[n]}')
            # print(f'Inhibitory covariance: {self.covs2[n]}')
            # for i in range(self.d):
            #     self.mean1[:, i] = my_mean1[i]
            #     self.mean2[:, i] = my_mean2[i]
            # responses = [self.rv1[n].pdf(xi) - self.rv2[n].pdf(xi) for xi in self.x_star]

             # my_mean1 = [2.5, 2.5]
        # my_mean2 = [.5, 1]
        # my_mean1 = [2.5, 2.5]
        # my_mean2 = [2.4, 2.6]
        
        # for i in range(self.d):
        #     self.mean1[:, i] = my_mean1[i]
        #     self.mean2[:, i] = my_mean2[i]

                # mean1[:,i] = self.min[i] + np.random.random(size=self.N) * self.scale[i] # min + random increment = mean
                # mean2[:,i] = self.min[i] + np.random.random(size=self.N) * self.scale[i] # min + random increment = mean
        # print(f"Excitatory mean: {self.mean1}")
        # print(f"Inhibitory mean: {self.mean2}")

        # for i in range(self.d):
        #         # covs[:,i,i] = (self.count[i]*1e8) * self.scale[i]**2 #np.random.random(size=self.N) * self.scale[i]
        #         # self.covs1[:,i,i] = np.random.random(size=self.N) * self.scale[i] / np.sqrt(self.count[i]) # diagonal elements of the covariance matrices
        #         # self.covs2[:,i,i] = np.random.random(size=self.N) * self.scale[i]**2/ np.sqrt(self.count[i])  #larger covariance, inhibitory
        #         desired_std = 1 # in stimulus units (Hz, dB, etc.)
        #         self.covs1[:,i,i] = desired_std**2  # e.g., 100
        #         self.covs2[:,i,i] = (desired_std*1.5)**2
        
        # print(self.covs1)
        # print(self.covs2)

    def sample(self, x, normalize = False):
            """
            Compute DoG response for all neurons at stimulus x (array-like, shape [d]).
            Returns shape [N].
            """
            self.z = np.zeros(self.N)
            for n in range(self.N):
                #self.z[n] = self.rv1[n].pdf(x) - self.rv2[n].pdf(x)
                self.z[n] = self.rv1[n].pdf(x) - self.rv2[n](x)
            
            if normalize:
                self.record_response(x, self.z, normalize=True)
            
            self.record_response(x, self.z)
            return self.z
    
    def verify_sln(self, peaks, n):
        ''' 
        Compares the predicted peak location to the true peak location 
        
        params:
        ----------
        peaks (list)  : predicted peak location (N,d)
        n (int)       : n-th neuron 

        returns:
        ----------
        dists (tuple) : predicted peaks - true peak of the n-th neuron
        count (int)   : a count of the number of correct predictions for each dimension
        mse (tuple)   : MSE between predicted peak and true peak of the n-th neuron
        '''
        # peaks shape: (N, d)
        dists = np.abs(peaks - self.peaks[n])           
        mse = mean_squared_error(peaks, self.peaks[n])
        count = np.count_nonzero(dists < self.tol)      # count dist within tolerance

        return dists, count, mse
    

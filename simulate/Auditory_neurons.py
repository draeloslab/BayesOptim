import numpy as np
import scipy.stats as stats
import pandas as pd
from sklearn.metrics import mean_squared_error
from simulate.neuron import Neuron
import matplotlib.pylab as plt

class Auditory_neurons(Neuron):
    ''' 
    Class to simulate O-shaped (Difference of Gaussian) neural tuning curves and responses to auditory stimuli (e.g. frequency and level)
    This models neurons with center excitation and lateral inhibition, such as those found in auditory pathways.
    Excitatory and inhibitory tuning curves are Gaussian-shaped, and the overall neuronal response is the difference
    between excitation and inhibition.
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

    def inhibition_sum(self, x, sb1, sb2): 
        """
        params:
        ----------
        x (tuple):  Coordinate(s) in stimulus space at which to evaluate the inhibition.
        sb1 : scipy.stats.multivariate_normal
            sideband 1 First inhibitory Gaussian distribution.
        sb2 : scipy.stats.multivariate_normal
            sideband 2 Second inhibitory Gaussian distribution.

        Returns
        -------
        float
            Sum of the two Gaussians' PDF values at x.
        """
        return sb1.pdf(x) + sb2.pdf(x)
        
    def gen_tuning_curves(self, type='indep', constraint = None):
        ''' 
        Generate tuning curves for all neurons from multivariate normal distributions and compute Difference of Gaussian response.
        Peak = Where DOG is maximal 
        There are three different kinds of tuning curves a neuron can be assigned randomly. Each have a different peak close to the center
        of the stimulus space.
        
        params:
        ----------
        type (str): "indep", "corr", "simulate"
        constraint (str): If not None, "linear"

        Returns
        -------
        None
        '''
        self.type = type
        self.rv1 = []  # List to store "excitation" random variable
        self.rv2 = []  # List to store "inhibition" random variable
        self.mean1 = np.zeros((self.N,self.d)) #smaller variance 
        self.covs1 = np.zeros((self.N,self.d,self.d))
        self.mean2 = np.zeros((self.N,self.d))  #larger variance
        self.covs2 = np.zeros((self.N,self.d,self.d))
    
        self.offset = 1
        center = np.array([(r - 1) / 2 for r in self.ranges]) #finding the center coordinate of the stimulus space
        sets = []

        # Set 0: Mean of the Excitatory sideband is centered in the stimulus space
        mean_exc1 = center.copy()
        cov_exc1 = np.eye(self.d) * (1.0 / self.d)
        cov_inh1 = np.eye(self.d) * (5.0 /self.d)

        # Set 1: excitatory center + inhibitory offset in dimension 0
        mean_exc2 = center.copy(); mean_exc2[0] += self.offset
        cov_exc2 = np.eye(self.d) * (1.0 / self.d)
        cov_inh2 = np.eye(self.d) * (3.0 /self.d)

        # Set 2: excitatory center + inhibitory offset in dimension 1
        mean_exc3 = center.copy()
        if self.d > 1:
            mean_exc3[1] += self.offset
            cov_exc3 = np.eye(self.d) * (1.0 / self.d)
            cov_inh3 = np.eye(self.d) * (4.0 /self.d)

        sets.append((mean_exc1, cov_exc1, cov_inh1))
        sets.append((mean_exc2, cov_exc2, cov_inh2))
        sets.append((mean_exc3, cov_exc3, cov_inh3))

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
            mean_exc, cov_exc, cov_inh = sets[idx]
            self.mean1[n] = mean_exc
            self.covs1[n] = cov_exc
            self.covs2[n] = cov_inh
            self.peak_types.append(idx)
            print(f"Neuron {n} is peak type {idx}")
            
            self.rv1.append(stats.multivariate_normal(mean=self.mean1[n], cov=self.covs1[n]))
            sideband1 = stats.multivariate_normal(mean= self.mean1[n]- self.offset, cov=self.covs2[n])
            sideband2 = stats.multivariate_normal(mean= self.mean1[n] + self.offset, cov=self.covs2[n])
            self.rv2.append((sideband1, sideband2)) # Store tuple of sidebands

            #Precompute DoG response for all points in self.x_star
            dog_response = []
            for xi in self.x_star:
                excitation = self.rv1[n].pdf(xi)
                inhibition = self.inhibition_sum(xi, sideband1, sideband2)
                dog_response.append(excitation - inhibition)
            dog_response = np.array(dog_response)
            self.responses.append(dog_response)

            peak_idx = np.argmax(self.responses[n])
            min_idx = np.argmin(self.responses[n])
            self.min[n] = self.x_star[min_idx]
            self.peaks[n] = self.x_star[peak_idx]
            #print(f"self.peaks: neuron {n} {self.peaks[n]}")


    def sample(self, x, addnoise = False, normalize = False):
            """
            Compute DoG response for all neurons at stimulus x (array-like, shape [d]).
            Returns shape [N].
            """
            seed = hash(tuple(x)) % (2**32)
            local_rng = np.random.default_rng(seed)
            self.z = np.zeros(self.N)
            for n in range(self.N):
                sb1, sb2 = self.rv2[n]
                inhibition = self.inhibition_sum(x, sb1, sb2)
                self.z[n] = self.rv1[n].pdf(x) - inhibition

                #if self.type == "indep_noise":
                if addnoise:
                # Choose your alpha (noise scaling)
                    alpha = 0.5   # You can tune this
                    noise_std = np.abs(self.z[n]) * alpha  # Scale noise with response magnitude
                    self.z[n] += local_rng.normal(0, noise_std)

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
        count = np.count_nonzero(dists < self.tol)   # count dist within tolerance

        return dists, count, mse
    

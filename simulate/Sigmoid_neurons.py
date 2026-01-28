import numpy as np
import scipy.stats as stats
import pandas as pd
from sklearn.metrics import mean_squared_error
from simulate.neuron import Neuron

class Sigmoid_neurons(Neuron):
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
        self.y = []
        
    def gen_tuning_curves(self, type='indep', constraint = None):
        ''' 
        Generates tuning curves for N number of neurons from a multivariate normal distribution with a random mean and covariance

        params:
        ----------
        type (str): "indep", "corr", "simulate"
        constraint (str): If not None, "linear"

        '''
        self.type = type
        self.rv = []  
        mean_sigmoid = np.zeros((self.N,self.d)) #smaller variance 
        covs_sigmoid = np.zeros((self.N,self.d,self.d))
        for i in range(self.d):
                mean_sigmoid[:,i] = self.min[i] + np.random.random(size=self.N) * self.scale[i] # min + random increment = mean
        for i in range(self.d):
                #covs[:,i,i] = (self.count[i]*1e8) * self.scale[i]**2 #np.random.random(size=self.N) * self.scale[i]
                covs[:,i,i] = np.random.random(size=self.N) * self.scale[i]**2 / np.sqrt(self.count[i]) # diagonal elements of the covariance matrices
    

        for n in range(self.N):
            self.rv1.append(stats.multivariate_normal(mean=mean1[n], cov=covs1[n]))
            self.rv2.append(stats.multivariate_normal(mean=mean2[n], cov=covs2[n]))
        #self.peaks = mean1.copy()

         # Create x_star from self.x
        xs = np.meshgrid(*self.x, indexing='ij')
        x_star = np.empty(xs[0].shape + (self.d,))
        for i in range(self.d):
            x_star[..., i] = xs[i]
        self.x_star = x_star.reshape(-1, self.d)

        # Find true DoG peaks using self.x_star
        self.peaks = np.zeros((self.N, self.d))
        for n in range(self.N):
            responses = [self.rv1[n].pdf(xi) - self.rv2[n].pdf(xi) for xi in self.x_star]
            peak_idx = np.argmax(responses)
            self.peaks[n] = self.x_star[peak_idx]

    def sample(self, x):
            """
            Compute DoG response for all neurons at stimulus x (array-like, shape [d]).
            Returns shape [N].
            """
            z = np.zeros(self.N)
            for n in range(self.N):
                z[n] = self.rv1[n].pdf(x) - self.rv2[n].pdf(x)
            self.record_response(x, z)
            return z

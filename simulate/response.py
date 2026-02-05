import numpy as np
from sklearn.metrics import mean_squared_error
from abc import ABC, abstractmethod

class Response:
    '''Response base class for Neuron'''
    
    def __init__(self, N, d):
         
         self.N = N # number of neurons, electrodes, etc.
         self.d = d # stimulus dimension
         self.resp_x = []
         self.resp_z = []

    
    def record_response(self, x, z, normalize=False):
            ''' 
            Records z response given an x sample

            params:
            ----------
            x (tuple)        : sample stimulus
            z                : simulated response given sample stimulus
            normalize (bool) : if True, normalize response by the maximum response

            '''
            
            #--z measured response
            self.resp_x.append(x)
            self.resp_z.append(z)

            if normalize:
                self.norm_responses()
        
    def norm_responses(self):
        ''' 
        Normalizes response by the maximum response
        '''
        max_response = np.max(self.resp_z)
        if max_response > 0:
            self.resp_z = [z / max_response for z in self.resp_z]

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
    
    def sample(self, x, *, record=True, **kwargs):

        x_sample = self._resolve_x(x, **kwargs)
        z = self._sample_impl(x_sample, **kwargs)

        if record:
             self.record_response(x_sample, z)

        return z
    
    def _resolve_x(self, x, **kwargs):
         
         return x
    
    @abstractmethod
    def _sample_impl(self, x_sample, **kwargs):
         
         raise NotImplemented
import numpy as np
from sklearn.metrics import mean_squared_error

class Neuron:
    ''' Base class for neurons '''

    def __init__(self, N, d, tol=1):

        self.N = N
        self.d = d
        self.tol = tol
        self.resp_x = []
        self.resp_z = []
        self.peaks = None
        self.covs = None
    
    def set_tuning_x(self, x):
        ''' 
        Scaling parameters for simulating tuning curves
        
        params:
        ----------
        x (list) : list of dimension ranges
        '''
        self.x = x
        self.scale = np.array([l[-1] - l[0] for l in self.x])
        self.min = np.array([l[0] for l in self.x])
        self.max = np.array([l[-1] for l in self.x])
        self.count = np.array([l.shape[0] for l in self.x])

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



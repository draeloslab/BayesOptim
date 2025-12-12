import yaml
import numpy as np
import pandas as pd

class Config():
    def __init__(self, file):
        self.params = self.read_config(file)
        self.__dict__.update(self.params)

    def read_config(self, file):
        with open(file, 'r') as file:
            parameters = yaml.safe_load(file)

        ## General
        np.random.seed(parameters['General']['seed']) #NOTE: need for improv

        ## Optimizers
        self.n = parameters['Optimizer']['n']  # number of optimizers #NOTE: need for improv 
        self.d = parameters['Stimuli']['d']     # number of dimensions in tuning curve

        ## Stimulus dimensions defined
        exs = []
        ranges = []
        for i in range(1,self.d+1):  # for each dimension
            string = 'x'+str(i)+'_range'
            length = parameters['Stimuli'][string]
            exs.append(np.arange(length))
            ranges.append(length)

        ## number of different optimizers
        self.optimizers = {}
        optim_keys = [k for k in parameters['Optimizer'] if k.startswith('optim_')]
        for optim_n in optim_keys:
            optimizer_info = parameters['Optimizer'][optim_n]
            # Extract optimizer name and kernels
            self.kernels = optimizer_info['kernel']  # Get the list of kernels
            stopping_crit = optimizer_info['stopping_crit']
            # Check if the number of kernels matches the dimension of the stimuli
            if len(self.kernels) != self.d:
                raise ValueError(f"{optim_n}) has mismatched kernel count. Expected {self.d}, got {len(self.kernels)}.")
            # Store optimizer configurations
            self.optimizers[optim_n] = {'kernels': self.kernels, "stopping_crit": stopping_crit}
            # only read in matern kernel 
            if "matern" in self.kernels:
                self.matern_nu = float(parameters['Optimizer'][optim_n]['matern_nu'])        # matern kernel smoothness 
            else:
                self.matern_nu = None
            # only read in rbf_periodic kernel
            if "rbf_periodic" in self.kernels:
                self.periodic_p =  float(parameters['Optimizer'][optim_n]['periodic_p'])
            else:
                self.periodic_p = None
        
        ## Stimulus space
        xs = np.meshgrid(*exs, indexing='ij')
        x_star = np.empty(xs[0].shape + (self.d,))
        for i in range(self.d):
            x_star[...,i] = xs[i]
        self.x_star = x_star.reshape(-1, self.d)
        
        print('Number of possible test points to optimize over: ', self.x_star.shape[0])

        ## GP parameters
        self.max = np.array([l[-1] for l in exs])
        self.gamma = float(parameters['Optimizer']['gamma']) / self.max    # 2e-1 * 1/SimPop.max
        self.var = float(parameters['Optimizer']['var'])                   # variance of kernel
        self.nu = float(parameters['Optimizer']['nu'])                # trade off explore exploit
        self.eta = float(parameters['Optimizer']['eta'])              # noise in GP
        
        ## initial test points
        self.init_T = parameters['General']['init_T']
        self.max_tests = parameters['General']['max_tests']
    
        parameters.update({'exs': exs, 'stim_choice': ranges})
        return parameters
    
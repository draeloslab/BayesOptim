import yaml
import os
import numpy as np
import pandas as pd
from simulate.sim_neurons import SimNeurons
from simulate.pseudo_neurons import PseudoNeurons
from utils.get_inputs import ExperimentalInputs

class Config():
    def __init__(self, file):
        self.params = self.read_config(file)
        self.__dict__.update(self.params)

    def read_config(self, file):
        with open(file, 'r') as file:
            parameters = yaml.safe_load(file)

        ## General
        np.random.seed(parameters['General']['seed'])

        self.data_folder = parameters['General']['data_folder']
        self.method = parameters['General']['method']

        ## Neurons
        self.N = parameters['Neurons']['N']     # number of neurons
        self.d = parameters['Stimuli']['d']     # number of dimensions in tuning curve
        if 'fr' in parameters['Neurons'].keys():
            fr = parameters['Neurons']['fr']

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

        ## Pseudo-neuron run
        if parameters['Neurons']['SimPop'] == 'pseudo':
            self.dataset = parameters['General']['dataset']

            pre_stim_window = parameters['Stimuli']['pre_stim_window']
            stim_extension = parameters['Stimuli']['stim_extension']
            stim_inputs = ExperimentalInputs(self.data_folder, self.dataset, pre_stim_window, stim_extension)  
            self.N = len(stim_inputs.C)
            self.x_stim = stim_inputs.x_stim

            if self.method == 'offline':
                self.response_data = stim_inputs.auc
            else:
                self.response_data = stim_inputs.C

            with open(os.path.join(self.data_folder, 'offline_sln', f'offline_sln_{self.dataset}_f.npy'), 'rb') as f:
                pred_means = np.load(f)

            pred_means = pred_means.reshape(self.N,ranges[0],ranges[1],ranges[2])
            peaks = []
            for i in range(pred_means.shape[0]):
                peaks.append(np.unravel_index(np.argmax(pred_means[i]), pred_means[i].shape))

            SimPop = PseudoNeurons(stim_inputs.C, stim_inputs.stimulus_df, stim_inputs.stim_conditions, peaks, pre_stim_window, stim_extension)
            SimPop.set_tuning_x(exs)

        ## Gaussian-simulated neurons
        if parameters['Neurons']['SimPop'] == 'gaussian':
            SimPop = SimNeurons(self.N, self.d, tol = 5*np.array([l[1]-l[0] for l in exs]))
            SimPop.set_tuning_x(exs)

            if 'tc_type' in parameters['Neurons'].keys():
                SimPop.gen_tuning_curves(type=parameters['Neurons']['tc_type'], constraint='linear')
            else:
                SimPop.gen_tuning_curves(type='indep', constraint='linear')

        xs = np.meshgrid(*exs, indexing='ij')
        x_star = np.empty(xs[0].shape + (self.d,))
        for i in range(self.d):
            x_star[...,i] = xs[i]
        self.x_star = x_star.reshape(-1, self.d)
        
        print('Number of possible test points to optimize over: ', self.x_star.shape[0])

        ## GP Parameters
        self.gamma = parameters['Optimizer']['gamma'] / SimPop.max    # 2e-1 * 1/SimPop.max
        self.var = parameters['Optimizer']['var']                     # variance of kernel
        self.nu = float(parameters['Optimizer']['nu'])                # trade off explore exploit
        self.eta = float(parameters['Optimizer']['eta'])              # noise in GP
        
        ## initial test points
        self.init_T = parameters['General']['init_T']
        self.max_tests = parameters['General']['max_tests']
        X0 = np.zeros((self.max_tests,self.d))
        for i in range(self.d):   
            rr = np.random.randint(0, high=ranges[i], size=(self.init_T,)) 
            X0[:self.init_T, i] = rr
        print('Initial test points: ', X0[:self.init_T])

        # we're using all previous sampled X's
        self.X0 = [X0[i].copy() for i in range(self.init_T)]
        x_index = self.init_T

        ## Generate initial sample response data
        for i in range(self.init_T):
            SimPop.sample(X0[i])  # Call sample only once
        y0 = np.zeros((self.max_tests, self.N))
        y0[:self.init_T,:] = np.array(SimPop.resp_z)[:self.init_T,:]
        self.y0 = [y0[i].copy() for i in range(self.init_T)] # similarly, we're using all y's
        parameters.update({'exs': exs, 'y0': self.y0, 'x_index': x_index, 'SimPop': SimPop})
        return parameters
import yaml
import os
import pickle
import numpy as np
from simulate.sim_neurons import SimNeurons
# from simulate.pseudo_neurons import PseudoNeurons
from simulate.bio_neurons import BioNeurons
from simulate.auditory_neurons import AuditoryNeurons
from optimizer import calc_offline_fit
from utils.get_inputs import ExperimentalInputs
from utils.gp_fit_utils import auc_clean_to_X_y
from utils.post_processing_utils import read_outliers_csv, stash_calibration_XY

class Config():
    def __init__(self, file):
        self.params = self.read_config(file)
        self.__dict__.update(self.params)

    def read_config(self, file):
        with open(file, 'r') as file_read:
            parameters = yaml.safe_load(file_read)

        ## General
        np.random.seed(parameters['General']['seed'])

        self.data_folder = parameters['General']['data_folder']
        self.method = parameters['General']['method']
        self.algorithm = parameters['General']['algorithm']
        self.offline_sln_path = os.path.join(self.data_folder, "offline_sln")  # ????
        if not os.path.exists(self.offline_sln_path):
            os.makedirs(self.offline_sln_path)
            print(f"Created directory for offline solutions at {self.offline_sln_path}")
        self.lab = parameters['General']['lab']
        
        ## Neurons
        self.N = parameters['Neurons']['N']     # number of neurons
        self.d = parameters['Stimuli']['d']     # number of dimensions in tuning curve
        if 'fr' in parameters['Neurons'].keys():
            fr = parameters['Neurons']['fr']
        s = parameters['Neurons']['add_noise']
        self.add_noise = None if str(s).lower() == 'none' else s
        self.noise_type = parameters['Neurons']['noise_type']  # 'uniform' or 'poisson'
        
        # new patch, added outliers
        try:
            outliers = read_outliers_csv(os.path.join('../aggregate_neuron_responses/output_var', self.lab, "outliers.csv"))[self.data_folder.split("/")[-1][7:]]
            print("These are outliers", outliers)
        except KeyError:
            print("No outliers found for this dataset.")
            outliers = None
        if os.path.isfile(os.path.join(self.data_folder, "analysis_stimY.pkl")) and os.path.isfile(os.path.join(self.data_folder, "analysis_stimX.pkl")):
            print("yes there are stimY and stimX")
            with open(os.path.join(self.data_folder, "analysis_stimY.pkl"), 'rb') as f:
                self.stimY = pickle.load(f)
            with open(os.path.join(self.data_folder, "analysis_stimX.pkl"), 'rb') as f:
                self.stimX = pickle.load(f)
            if len(self.stimX[0]) > self.d: # assuming d = 5
                print(f"stimX has more than 5 dimensions, likely {len(self.stimX[0])}d trimming down to 5")
                self.stimX, self.stimY = stash_calibration_XY(self.data_folder, self.stimX, self.stimY)
        else:
            self.stimY = None
            self.stimX = None

       ## Stimulus dimensions defined
        exs = []
        ranges = []
        self.gamma_single = parameters['Optimizer']['gamma']
        for i in range(1,self.d+1):  # for each dimension
            string = 'x'+str(i)+'_range'
            length = parameters['Stimuli'][string]
            exs.append(np.arange(length))
            ranges.append(length)
        self.exs = exs

        ## optimizers
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

            pre_stim_window = parameters['Stimuli']['pre_stim_window']
            stim_extension = parameters['Stimuli']['stim_extension']
            stim_inputs = ExperimentalInputs(self.data_folder, "", pre_stim_window, stim_extension, self.lab, self.N, self.stimY, outliers)  
            self.x_stim = stim_inputs.x_stim
            self.stim_df = stim_inputs.stimulus_df
            self.good_neuron_idx_ls = stim_inputs.good_neuron_idx_ls
            self.N = len(self.good_neuron_idx_ls)
            print("Updated number of neurons (N) to match stimulus inputs: ", self.N)

            if self.method == 'offline':
                self.response_data = stim_inputs.auc
            else:
                self.response_data = stim_inputs.C
                self.auc = stim_inputs.auc
            try:
                gamma_str = str(self.gamma_single).replace('.', '')
                file_name = f'offline_sln_f_g{gamma_str}.npy'
                    # change from number coding to gamma coding for file name
                full_path = os.path.join(self.offline_sln_path, file_name)
                print(file_name, full_path)
                print(f"stimY length {len(self.stimY)}")
                print(f"stimX length {len(self.stimX)}")
                try:
                    with open(full_path, 'rb') as f:
                        pred_means = np.load(f)
                        if pred_means.shape[0] > self.N:
                            pred_means = pred_means[self.good_neuron_idx_ls]
                            print(f"trimming down offline solution to {self.N} of neurons. Current shape of pred_means is {pred_means.shape}")
                        elif pred_means.shape[0] == self.N:
                            pass
                        else:
                            raise ValueError(f"Number of neurons in offline solution ({pred_means.shape[0]}) is smaller than N ({self.N}). Please check again")
                except FileNotFoundError as e:
                    print("offline file does not exist. Rerunning offline fitting")
                    config_test = Config_get_params(file, gamma_new=self.gamma_single, stop_new=stopping_crit)
                    print(config_test.gamma)

                    # if not os.path.exists(full_path):
                    if isinstance(self.stimX, list):
                        stimXs = np.squeeze(np.array(self.stimX), axis=-1)
                    elif isinstance(self.stimX, np.ndarray):
                        stimXs = self.stimX
                    else:
                        raise TypeError(f"stimX is of unsupported type: {type(self.stimX)}")
                    X_all, y_all = auc_clean_to_X_y(self.auc, stimXs)

                    
                    try:
                        # f_offline, sigma_offline = calc_offline_fit(np.asarray(X), np.asarray(y), config, kernels)
                        pred_means, _ = calc_offline_fit(np.asarray(X_all), np.asarray(y_all), config_test, config_test.optimizers['optim_1']['kernels']) #X_first_100[0]
                    except ValueError:
                        print("Value Error, X and y may need to be dtype=object")
                        pred_means, _ = calc_offline_fit(np.asarray(X_all,dtype=object), np.asarray(y_all,dtype=object), config_test, config_test.optimizers['optim_1']['kernels'])
                    np.save(full_path, pred_means)
                    print(f"saving offline results to {full_path}")
                    
                    # pass
                if self.d == 4:
                    pred_means = pred_means.reshape(self.N,ranges[0],ranges[1],ranges[2], ranges[3])  # TODO: uncomment this after testing
                elif self.d == 3:
                    pred_means = pred_means.reshape(self.N,ranges[0],ranges[1],ranges[2])  # TODO: comment this out, this is for early burgess
                elif self.d == 5:

                    pred_means = pred_means.reshape(self.N,ranges[0],ranges[1],ranges[2], ranges[3], ranges[4])
                elif self.d == 2:
                    pred_means = pred_means.reshape(self.N,ranges[0],ranges[1],ranges[2])
                peaks = []
                for i in range(pred_means.shape[0]):
                    peaks.append(np.unravel_index(np.argmax(pred_means[i]), pred_means[i].shape))
                SimPop = BioNeurons(stim_inputs.C, stim_inputs.stimulus_df, stim_inputs.stim_conditions, peaks, pre_stim_window, stim_extension, pred_means, self.stimY, good_neuron_idx_ls=self.good_neuron_idx_ls)
                SimPop.set_tuning_x(exs, ranges)
            except Exception as e:  
                print(f"An error occurred: {e}")

        ## Gaussian-simulated neurons
        if parameters['Neurons']['SimPop'] == 'gaussian':
            SimPop = SimNeurons(self.N, self.d, tol = 5*np.array([l[1]-l[0] for l in exs]), add_noise=self.add_noise, noise_type=self.noise_type)
            SimPop.set_tuning_x(exs, ranges)

            if 'tc_type' in parameters['Neurons'].keys():
                SimPop.gen_tuning_curves(type=parameters['Neurons']['tc_type'], constraint='linear')
            else:
                SimPop.gen_tuning_curves(type='indep', constraint='linear')
        

        #Auditory Difference of Gaussian-simulated neurons
        if parameters['Neurons']['SimPop'] == 'auditory':
            SimPop = AuditoryNeurons(self.N, self.d, tol = np.array([l[1]-l[0] for l in exs]))
            SimPop.set_tuning_x(exs, ranges)
            SimPop.gen_tuning_curves(type='indep', constraint=None)

        xs = np.meshgrid(*exs, indexing='ij')
        x_star = np.empty(xs[0].shape + (self.d,))
        for i in range(self.d):
            x_star[...,i] = xs[i]
        self.x_star = x_star.reshape(-1, self.d)
        
        print('Number of possible test points to optimize over: ', self.x_star.shape[0])

        ## GP Parameters
        # self.gamma = parameters['Optimizer']['gamma'] / SimPop.max    # 2e-1 * 1/SimPop.max
        self.var = parameters['Optimizer']['var']                     # variance of kernel
        self.nu = float(parameters['Optimizer']['nu'])                # trade off explore exploit
        self.eta = float(parameters['Optimizer']['eta'])              # noise in GP
        self.mse_cutoff = float(parameters['Optimizer']['mse_cutoff']) 
        
        ## initial test points
        self.init_T = parameters['General']['init_T']
        self.max_tests = parameters['General']['max_tests']
        if SimPop is not None:
            self.gamma = self.gamma_single / SimPop.max #* SimPop.max  #/ SimPop.max  # 2e-1 * 1/SimPop.max
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

class Config_get_params():
    def __init__(self, file, gamma_new = None, stop_new = None):
        print("inside config_get_params!!")
        self.gamma_new = gamma_new
        self.stop_new = stop_new
        self.params = self.read_config(file)
        self.__dict__.update(self.params)
    
    def read_config(self, file):
        with open(file, 'r') as file:
            parameters = yaml.safe_load(file)
        print("got the parameters")
        ## General
        np.random.seed(parameters['General']['seed']) #NOTE: need for improv

        ## Optimizers
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
            # optimizer_name = optimizer_info['name']  # Get the optimizer name (e.g., "Optimizer")
            kernels = optimizer_info['kernel']  # Get the list of kernels
            stopping_crit = optimizer_info['stopping_crit']
            # Check if the number of kernels matches the dimension of the stimuli
            if len(kernels) != self.d:
                raise ValueError(f"{optim_n}) has mismatched kernel count. Expected {self.d}, got {len(kernels)}.")
            # Store optimizer configurations
            self.optimizers[optim_n] = {'kernels': kernels, "stopping_crit": stopping_crit}
            # only read in matern kernel 
            if "matern" in kernels:
                self.matern_nu = float(parameters['Optimizer'][optim_n]['matern_nu'])        # matern kernel smoothness 
            else:
                self.matern_nu = None

            if "rbf_periodic" in kernels:
                self.periodic_p =  float(parameters['Optimizer'][optim_n]['periodic_p'])
                print(f"Using periodic kernel with period {self.periodic_p}")
            else:
                self.periodic_p = None
        
        ## Stimulus space
        xs = np.meshgrid(*exs, indexing='ij')
        x_star = np.empty(xs[0].shape + (self.d,))
        for i in range(self.d):
            x_star[...,i] = xs[i]
        self.x_star = x_star.reshape(-1, self.d)
        
        # print('Number of possible test points to optimize over: ', self.x_star.shape[0])

        ## GP parameters
        self.max = np.array([l[-1] for l in exs])
        if self.gamma_new is not None:
            print(f"Config_get_params() using new gamma {self.gamma_new}")
            self.gamma = self.gamma_new / self.max
        else:
            self.gamma = float(parameters['Optimizer']['gamma']) / self.max    # 2e-1 * 1/SimPop.max
        self.var = float(parameters['Optimizer']['var'])                   # variance of kernel
        self.nu = float(parameters['Optimizer']['nu'])                # trade off explore exploit
        self.eta = float(parameters['Optimizer']['eta'])              # noise in GP
        ## initial test points
        self.init_T = parameters['General']['init_T']
        self.max_tests = parameters['General']['max_tests']
    
        parameters.update({'exs': exs, 'stim_choice': ranges})
        print()
        return parameters
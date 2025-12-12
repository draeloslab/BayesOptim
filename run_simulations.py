import sys
import numpy as np
import os
import yaml
from datetime import datetime

from model.bayesopt_sampling import bayesopt_sampling 
from model.random_sampling import random_sampling
from model.optimizer import calc_offline_fit 
from plot import plot_correct_prediction, plot_peak_value, plot_run_time, plot_tuningcurves_eval, plot_acqf, plot_mse
from utils.save_results import save_results

# Choosing what type of algorithm to run (simulation, improv, etc.)
if len(sys.argv) > 1:
    parameter_file = sys.argv[1]
else:
    parameter_file = input('Enter parameter file: ')

param_file_path = os.path.join('parameters_all', parameter_file)
with open(param_file_path, 'r') as file:
    parameters = yaml.safe_load(file)

# Load in parameters
# config file will expect a dataset input argument if pseudo_neuron is required 
if parameters['General']['algorithm'] == 'sim':
    from model.sim_config import Config

    config = Config(file=param_file_path)

elif parameters['General']['algorithm'] == 'improv':
    from model.improv_config import Config
    #NOTE: this section is more for testing/debugging the improv config (should not run any optimization)

    config = Config(file=param_file_path)
    print("Improv config parameters")
    print(config)
    sys.exit()


optimizers = list(config.optimizers.keys())

# Evaluation method
if len(sys.argv) > 2:
    eval_method = sys.argv[2]
else:
    eval_method = None

while True:
    if config.algorithm == 'sim':

        if config.method == 'offline':
            # This method calculates the offline fit of the GP. Used for comparing experimental data tuning curves
            f, sigma = calc_offline_fit(config.x_stim, config.response_data, config.x_star, config.var, config.gamma, config.eta, config.kernels)
            with open(config.offline_sln_path + '_f.npy', 'wb') as file:
                np.save(file,f)
            with open(config.offline_sln_path + '_sigma.npy', 'wb') as file:
                np.save(file,sigma)

            sys.exit()

        if config.method == 'online':
        
            if eval_method == 'eval':
                results_dict = bayesopt_sampling(config, print_flag=True)
                # plot tuning curves 
                N = config.N
                exs = config.exs
                SimPop = config.SimPop
                plot_tuningcurves_eval(N, exs, SimPop, candidates=results_dict['loc_list'], method='simulate')
                # plot_stopping(stopping_allN)
                # plot_acqf(acq_list)
                break

            elif eval_method is None or eval_method == ' ':
                results_dict = bayesopt_sampling(config, print_flag=True)

                break

            else:
                print('Chosen evaluation method is not in list of options.')
                eval_method = input('Enter the evaluation method (Blank for none or eval): ')

        elif config.method == 'parallel':
            # what are the difference for sharedmem vs. backend loky? 
            # thread-based vs process-based backend
            results = Parallel(n_jobs=len(optimizers),  require='sharedmem')(  # backend='loky', require='sharedmem'
                delayed(bayesopt_sampling_parallel)(config, optimizer, print_flag=False) 
                for optimizer in optimizers)
            
            # Extract results
            Pr_list_all = []
            mse_final_all = []
            loc_list_all = []
            max_allN_all = []
            stopping_allN_all = []
            runt_list_all = []

            for result in results:
                # TODO: update this to reflect the changes in parallel_joblib
                results_dict = result
                # any quicker way to do stuff?
                Pr_list_all.append(results_dict['Pr_list'])
                mse_final_all.append(results_dict['mse_final'])
                loc_list_all.append(results_dict['loc_list'])
                max_allN_all.append(results_dict['max_allN'])
                stopping_allN_all.append(results_dict['stopping_allN'])
                runt_list_all.append(results_dict['runt_list'])
            
            if eval_method == 'eval':
                # plot tuning curves 
                # plot_tuningcurves_eval(N, exs, SimPop, candidates=loc_list, method='manual')
                plot_stopping(results_dict['stopping_allN'])
                # plot_acqf(acq_list)
                break

            elif eval_method is None or eval_method == ' ':
                break

            else:
                print('Chosen evaluation method is not in list of options.')
                eval_method = input('Enter the evaluation method (Blank for none or eval): ')

    else:
        config.algorithm = None

rsPr_list = random_sampling(config, print_flag=True)

if len(optimizers) > 1 and config.method == "parallel":  # not sure about this
    for i in range(len(optimizers)):
        plot_correct_prediction(N=config.N, Pr_list=Pr_list_all[i], rsPr_list=rsPr_list, optimizer_name=optimizers[i])
        plot_peak_value(x1=config.exs[0], x2=config.exs[1], mse_final=mse_final_all[i], loc_list=loc_list_all[i], SimPop=config.SimPop)
        plot_mse(mse_final=mse_final_all[i])
else:  # running single kernel
    plot_correct_prediction(N=config.N, Pr_list=results_dict['Pr_list'], rsPr_list=rsPr_list, optimizer_name=optimizers[0])
    plot_peak_value(x1=config.exs[0], x2=config.exs[1], mse_final=results_dict['mse_final'], loc_list=results_dict['loc_list'], SimPop=config.SimPop, optimizer_name=optimizers[0])
    plot_mse(mse_final=results_dict['mse_final'], optimizer_name=optimizers[0])
    plot_run_time(test_time_neuron=results_dict['test_time_neuron'], average=True)

if parameters['General']['save_results'] == True:
    save_results(param_file_path=param_file_path, results_dict=results_dict)
    

    





import sys
import time
import yaml
from datetime import datetime
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import jax
import jax.numpy as np

from plot import plot_correct_prediction, plot_peak_value, plot_run_time, plot_acqf, plot_mse#, plot_tuningcurves_eval
from utils.save_results import save_results
from model.bayesopt_sampling import bayesopt_sampling 
from model.kernel import kernel_rbf
from model.kernel import kernel_rbf_periodic
from model.kernel import kernel_matern
from model.kernel import kernel
from model.sim_config import Config

start = time.time()

parameter_file = "parameters_indep.yml"
param_file_path = os.path.join('parameters', parameter_file)
with open(param_file_path, 'r') as file:
    parameters = yaml.safe_load(file)

config = Config(file=param_file_path)

results_dict = bayesopt_sampling(config, print_flag=True)

end = time.time()
print('Elapsed time:', end-start, 's')

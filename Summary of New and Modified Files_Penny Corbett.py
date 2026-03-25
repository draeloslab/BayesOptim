'''
Simulate the auditory neurons using the model.sim_config.py 
New scripts I added:
simulate.Auditory_neurons.py 
    Class to simulate O-shaped (Difference of Gaussian) neural tuning curves in response to auditory stimuli (e.g. frequency and level
    as measured for Frequency Response Area)

for ground truth plots of these Auditory neuron tuning curves (three types), 
    plot_tuningcurves(N, SimPop, config) (found in plot.py, called in run_simulations)
to check with sampling plot:
    plot_tuningcurves_sampled(neuron_num, config, f_peak = None) (also found in plot.py)

model.grid_sample.py 
    the center of the stimulus space is finely sampled, note that x_coarse and x_fine have to be adjusted manually
    for a dimension size beyond d1 = 10, d2 = 10

utils.pareto_plot.py

utils.plot_noisy_and_clean_tuning

utils_plotting_offline_and_online

utils.stop_functions_plots.py   
    plots the stopping functions: takes the stopping value (Expected Improvement value from the results_dict output 
    from bayesopt_sampling for each test for each neuron, allows a comparison of different runs of the simulation 
    as a function of gamma in one plot and nu in another plot

Modified model.bayesopt_sampling.py to have a second way of counting a neuron as "correct"
    Pr_list_correct_solution: probabilities of correct predictions even if EI is still above stopping criteria 
    using Bayes Opt sampling
    Pr_list: both correct peak AND EI < stopping_crit

Modified parameters_indep.yml to include a mse_cutoff value, so that random and grid sampling scripts can
follow the same stopping requirements


'''
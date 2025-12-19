import time
import numpy as np
from model.optimizer import Optimizer  # all of them


# @profile
def bayesopt_sampling(c, print_flag=False):
    ''' Bayesian optimization

    For each neuron, a Gaussian Process (GP) is initialized and then an iterative process of updating the GP in order to minimize
    the objective function (Expected Improvement) while also satisifying the condition of predicting a "close enough" tuning curve
    peak location. For each neuron, there is a max_tests number of iterations that is available to update the GP. If the stopping 
    conditions are not satisifed by the end of the maximum number of iterations, the neuron index is stored and is rerun after the 
    remaining neurons' peak locations have been predicted. 

    params:
    ---------
    c (dict)                : contains parameters from Config file
    print_flag (bool)       : if True, will print log

    returns: 
    ---------
    Pr_list (list)          : probabilities of correct predictions using Bayes Opt sampling
    mse_final (list)        : final MSE values for each neuron
    loc_list (list)         : final predicted peak locatoins for each neuron
    max_allN (list)         : predicted peak locations for each test run for each neuron
    stopping_allN (list)    : stopping value (Expected Improvement value) for for each test for each neuron
    runt_list (list)        : run time for each neuron
    test_time_neuron (list) : run time for each test for each neuron
    '''

    t=0
    cn=0
    nt=0

    printing = print_flag


    ## TODO: results object
    Pr_list = []
    mse_allN = [None]*c.N
    stopping_allN = [None]*c.N
    max_allN = [None]*c.N
    loc_list = [None]*c.N
    runt_list=[0]*c.N
    # neuron_time = [0]*c.N
    test_time_neuron = [[0]*c.max_tests]*c.N
    rerun_neurons = []
    f_all = [None]*c.N
    sigma_all = [None]*c.N
    sample_x = [None]*c.N
    sample_y = [None]*c.N

    print('Running Bayesian Optimization ...')

    n_optim = -1
    flag_all_neurons = False    # Flag to say if we have gone through all neurons
    rerun_flag = False          # If True, there are neurons we need to return to and try again
    torun_list = []             # The list that stores the neurons'optim(index) to re-run 
    optim_n = list(c.optimizers.keys())
    if c.params['General']['algorithm'] != "parallel":
        assert len(optim_n) == 1, "Not running BayesOpt in parallel, only accept one optimizer; \
            comment out optimizers not being used"
        optim_n = optim_n[0]
        # optimizer_class = 'Optimizer'#optimizer_class[0]
    optimizer_kernel = c.optimizers[optim_n]['kernels']
    optimizer_stopping_crit = c.optimizers[optim_n]['stopping_crit']
    optimizer_class = Optimizer #globals().get(optimizer_class) # got rid of the dynamic instantiation
    print("You're using these kernels:", optimizer_kernel, ". The stopping crit is:", optimizer_stopping_crit)
    print("Gamma: {}; Nu: {}; var: {}; eta: {}; matern nu (if any) {}".format(c.gamma, c.nu, c.var, c.eta, c.matern_nu))
    while not flag_all_neurons: 
        
        ## Set up which neuron is going to be optimized, or if we are done. 
        if (rerun_flag and len(torun_list) == 0) or (len(torun_list) == 0 and n_optim == c.N-1): 
            # Finished rerun or no unoptimized neurons
            flag_all_neurons = True
            # print("first if")
            break
        if len(torun_list) != 0 and n_optim == c.N-1: 
            # One-time check if there are some neurons to re-run once normal operation is done 
            # print("second if")
            rerun_flag = True
        if len(torun_list) != 0 and rerun_flag:
            n_optim = torun_list.pop(0)  
            # print("third if")
            print('Rerunning neuron #: {}'.format(n_optim))
            rerun_neurons.append(n_optim)
        else:
            # print("fourth if")
            # Move on to next neuron
            n_optim += 1 


        ## Optimize this neuron
        correct_solution = False
        done_optimizing = False
    
        # start_neuron = time.time()
        if printing:
            print('Number ', n_optim, '; peak of this neuron: ', c.SimPop.peaks[n_optim])

        optim = optimizer_class(c, optimizer_kernel)
        

        # with types; uncomment if confident about the double-peak detection    
        # optim = optimizer_class(c.gamma, c.var, c.nu, c.eta, c.x_star, optimizer_kernel, c.params['Neurons']['tc_type'])
        
        # the following twos are the original ones
        # optim = Optimizer(c.gamma, c.var, c.nu, c.eta, c.x_star, c.kernels) 
        # optim = Optimizer_linear(c.gamma, c.var, c.nu, c.eta, c.x_star, c.kernels)
        
        
        X_train = np.array(c.X0)
        y_train = np.array(c.y0)[:, n_optim]
        optim.initialize_GP(X_train, y_train)
        initial_X = X_train.copy()
        initial_y = y_train.copy()
        selected_X = []
        selected_y = []
       
        stopping_list = []
        max_list = []
        mse_list = []
        f_list = []
        sigma_list = []
        test_time = []
        
        for cnt in range(c.max_tests):
            start_test = time.time()
            nt += 1
            t += 1
            Pr_list.append(cn/c.N)

            _, xt_1 = optim.max_acq()  # change this to multiple peaks? 
            y = c.SimPop.sample(xt_1)

            c.X0.append(xt_1)
            c.y0.append(y)
            c.x_index += 1

            selected_X.append(xt_1)
            selected_y.append(y[n_optim])
            # print(n_optim, xt_1, y[n_optim])

            optim.update_GP(xt_1, y[n_optim])
            pl = optim.x_star[np.argmax(optim.f)]
            dists, pixels_correct, mse = c.SimPop.verify_sln(pl, n_optim)
            # dists, pixels_correct, mse = c.SimPop.verify_sln_double(pl, n_optim)
            EI, PI = optim.stopping()
            mse_list.append(mse)
            stopping_list.append(optim.stopping())
            max_list.append(pl)
            f_list.append(optim.f)
            sigma_list.append(optim.sigma)

            # if pixels_correct > (c.d-1) and not correct_solution:
            #     correct_solution = True
                # print("for first if")

            # if EI < optimizer_stopping_crit and correct_solution and not done_optimizing:  # got rid of the correct_solution, more like improv
            #if optim.stopping() < c.stopping_crit and correct_solution and not done_optimizing:  # the original original one
            if EI < optimizer_stopping_crit and not done_optimizing:
                done_optimizing = True
                cn += 1
                runt_list[n_optim] += nt
                # print("for second if")
                break

            if cnt == c.max_tests-1:
                print('Used all ', c.max_tests, ' tests and did not finish; got close? ', np.around(dists, 2))
                runt_list[n_optim] += nt
                # print("for thrid if")
                # end_neuron = time.time()
                if n_optim not in rerun_neurons:
                    torun_list.append(n_optim)
        #print(torun_list)
        if rerun_flag:  
            # Append and update the original data record
            try:
                mse_allN[n_optim].extend(mse_list)       
                max_allN[n_optim].extend(max_list)
                loc_list[n_optim] = max_list[-1]
                stopping_allN[n_optim].extend(stopping_list)
                f_all[n_optim].extend(f_list)
                sigma_all[n_optim].extend(sigma_list)
                sample_x[n_optim]['initial'].extend([initial_X])
                sample_y[n_optim]['initial'].extend([initial_y])
                sample_x[n_optim]['selected'].extend(selected_X)
                sample_y[n_optim]['selected'].extend(selected_y)
            except:
                breakpoint()
        else:
            # Trace the mse of single neuron optimization
            mse_allN[n_optim] = mse_list     
            max_allN[n_optim] = max_list
            loc_list[n_optim] = max_list[-1]
            stopping_allN[n_optim] = stopping_list
            f_all[n_optim] = f_list
            sigma_all[n_optim] = sigma_list
            sample_x[n_optim] = {
                'initial': [initial_X],
                'selected': selected_X
            }
            sample_y[n_optim] = {
                'initial': [initial_y],
                'selected': selected_y
            }

        test_time_neuron[n_optim] = test_time
        # neuron_time[n_optim] = end_neuron - start_neuron    
    Pr_list.append(cn/ c.N)
    mse_final = [neuron_mse[-1] for neuron_mse in mse_allN]

    results_dict = {
        "Pr_list": Pr_list,
        "mse_final": mse_final,
        "loc_list": loc_list,
        "max_allN": max_allN,
        "stopping_allN": stopping_allN,
        "runt_list": runt_list,
        "f_all": f_all,
        "sigma_all": sigma_all,
        "sample_x": sample_x,
        "sample_y": sample_y,
        "test_time_neuron": test_time_neuron,
    }
    
    return results_dict

import numpy as np
from sklearn.metrics import mean_squared_error

def random_sampling(c, print_flag=False):
    ''' Random sampling

    params:
    ---------
    c (dict)          : contains parameters from Config file (i.e. N, d, max_tests, X0, x_star, SimPop)
    print_flag (bool) : if True, will print log

    returns: 
    ---------
    rsPr_list (list)  : probabilities of correct predictions using random sampling
    '''

    rsnt=0 # overall # of tests for an algorithm run\
    rscn=0 # num of correct neurons in random sampling
    t=0
    rsPr_list=[] # random sampling Pr_list; num of pred neurons / overall neurons for random sampling

    ## Making local variables here for simplicity
    N = c.N
    d = c.d
    max_tests = c.max_tests
    X0 = c.X0
    x_star = c.x_star
    SimPop = c.SimPop

    printing = print_flag

    print('Running Random Sampling ...')   
    for n_optim in range(N):
                
        #################Optimize per neuron#############################
        # For each neuron, X0 contains the initial X + points sampled from X_* from previous neurons' optimization.
        #i.e., T is enlarging and thus A shape is bigger

        stopping_list = []
        max_list = []
        mse_list = []
        flag = False
        myflag = False
        max_value = 0
        peak_guess = X0[0]

        xs_copy=x_star.copy() # copy a new X_star for each neuron

        if printing:
            print('Number ', n_optim, '; peak of this neuron: ', SimPop.peaks[n_optim])

        # for double-peaks only
        random_peak_number = np.random.choice([0,1])
        for cnt in range(max_tests):
            rsnt+=1    # nt +1 each run each neuron
            t+=1
            rsPr_list.append(rscn/N) # num of pred neurons / overall neurons for random sampling

            nrows = xs_copy.shape[0]
            random_indices = np.random.choice(nrows, size=1, replace=False)
            pl= xs_copy[random_indices,] # the loc of peak
            pl=pl.ravel()
            # for double-peaks only
            if c.params['Neurons']['tc_type'] == "double_peaks":
                dists = np.abs(pl - SimPop.peaks[random_peak_number][n_optim])
                mse= mean_squared_error(pl,  SimPop.peaks[random_peak_number][n_optim])
            else:  # for unique peaks
                dists = np.abs(pl - SimPop.peaks[n_optim])
                mse= mean_squared_error(pl,  SimPop.peaks[n_optim])

            # # for double-peaks only
            # dists = np.abs(pl - SimPop.peaks[random_peak_number][n_optim])
            # mse= mean_squared_error(pl,  SimPop.peaks[random_peak_number][n_optim])

            count = np.count_nonzero(dists < SimPop.tol) #--count dist within tolerance
            xs_copy=np.delete(xs_copy, random_indices, axis=0)#sample without replacement

            if count > (d-1) and not flag:
                flag = True


            if mse < 2 and flag and not myflag: #8e-11:  #0.2
                myflag = True
                rscn+=1
                break

            if cnt == max_tests-1:
                print('-------------------- used all ', max_tests,' tests and did not finish; got close? ', np.around(dists, 2))

    rsPr_list.append((rscn)/N)

    return rsPr_list
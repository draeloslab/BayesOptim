import numpy as np
from sklearn.metrics import mean_squared_error

def grid_sampling(c, print_flag=False):
    ''' Random sampling

    params:
    ---------
    c (dict)          : contains parameters from Config file (i.e. N, d, max_tests, X0, x_star, SimPop)
    print_flag (bool) : if True, will print log

    returns: 
    ---------
    gsPr_list (list)  : probabilities of correct predictions using grid sampling in which the center of the grid
    is more finely sampled than the rest of the stimulus space
    '''

    gsnt=0 # overall # of tests for an algorithm run
    gscn=0 # num of correct neurons in random sampling
    t=0
    gsPr_list=[] # grid sampling Pr_list; num of pred neurons / overall neurons for random sampling

    N = c.N
    d = c.d
    max_tests = c.max_tests
    X0 = c.X0
    x_star = c.x_star
    SimPop = c.SimPop
    x_coarse = np.array([0, 1, 2, 3, 6, 7, 8, 9])
    x_fine   = np.array([4, 4.5, 5, 5.5])  # finer around middle of stimulus space
    x_range = np.concatenate([x_coarse, x_fine])
    y_range = np.arange(0, 10, 0.5)
    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
    grid_points = np.column_stack([X.ravel(), Y.ravel()]) 

    printing = print_flag

    print('Running Grid Sampling ...')   
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
        xs_copy = grid_points.copy()  #includes more points to sample around the center of the stimulus space
        
        if printing:
            print('Number ', n_optim, '; peak of this neuron: ', SimPop.peaks[n_optim])


        for cnt in range(max_tests):
            gsnt+=1    # nt +1 each run each neuron
            t+=1
            gsPr_list.append(gscn/N) # num of pred neurons / overall neurons for random sampling

    
            random_indices = np.random.choice(len(xs_copy), size=1, replace=False)
            pl= xs_copy[random_indices,] # the loc of peak
            pl=pl.ravel()
            dists = np.abs(pl - SimPop.peaks[n_optim])
            mse= mean_squared_error(pl,  SimPop.peaks[n_optim])

            count = np.count_nonzero(dists < SimPop.tol) #--count dist within tolerance
            xs_copy=np.delete(xs_copy, random_indices, axis=0)#sample without replacement

            if count > (d-1) and not flag:
                flag = True


            #if mse < .05 and flag and not myflag: #8e-11:  #0.2
            if mse <= c.mse_cutoff and flag and not myflag: #8e-11:  #0.2
                myflag = True
                gscn+=1
                break
        

            if cnt == max_tests-1:
                print('-------------------- used all ', max_tests,' tests and did not finish; got close? ', np.around(dists, 2))

    gsPr_list.append((gscn)/N)
    return gsPr_list
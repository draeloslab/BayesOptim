import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
from model.kernel import kernel

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Optimizer():
    def __init__(self, c, kernels): # gamma, var, nu, eta, x_star, kernels, matern_nu
        self.c = c
        self.variance = c.var # var f
        self.nu = c.nu
        self.eta = c.eta
        self.x_star = c.x_star

        self.d = self.x_star.shape[1] # d dimensions

        self.kernels = kernels

        self.f = None
        self.sigma = None       ## Note: this is actually sigma squared
#       self.sigma = np.diagonal(self.sigma); True sigma
        self.kvv=None # Kvv=K(X*,X*)=sigma_2
        self.X_t = None
        self.K_t = None
        self.k_star = None
        self.y = None
        self.A = None
        self.test_pt = None

        self.t = 0
        # self.types = types # double_peak, linear_uvn, indep etc.


    #@profile
    def initialize_GP(self, X, y):
        ##Create initial parameters
        ## X is a matrix (T,d) of initial T measurements we have results for

        # self.mu = 0


        self.X_t = X

        self.y = y #1D arr (T,)
        T = self.X_t.shape[0]
        a = self.x_star.shape[0]

        self.test_count = np.zeros(a) # 1D array with len=a
        # test_count= list: an element indicates # count of a point

        self.K_t = kernel(self.X_t, self.X_t, self.c, self.kernels)#kernel(self.X_t, self.X_t, self.variance, self.gamma, self.kernels, self.matern_nu) #kernel_rbf(self.X_t, self.X_t,self.variance,self.gamma)    
        self.k_star = kernel(self.X_t, self.x_star,self.c, self.kernels)# self.variance,self.gamma, self.kernels, self.matern_nu #kernel_rbf(self.X_t, self.x_star,self.variance,self.gamma)
        self.kvv= kernel(self.x_star, self.x_star, self.c, self.kernels) # self.variance,self.gamma, self.kernels, self.matern_nu #kernel_rbf(self.x_star, self.x_star,self.variance,self.gamma)

        self.A = np.linalg.inv(self.K_t + self.eta**2 * np.eye(T)) # A^(-1) in fact
    
        #T (the number of rows in self.K_t); inverse

        self.f = self.k_star.T @ self.A @ self.y  #1D arr (V,)
        # @ matrix dot

        self.sigma = self.variance * np.eye(self.x_star.shape[0]) - self.k_star.T @ self.A @ self.k_star
        self.t = T

    #@profile
    def update_obs(self, x, y):
        # Create y_t1, x_t1 for updation

        self.y_t1 = np.array([y]) #[] add 1 extra dim to scalar y:scalar - 1D
        #1D arr (1, )
        
        #objects y(t+1),x(t+1) for update
        self.x_t1 = x[None,...]
        #1d arr (1,d)

        #[None,...] indexing is used to add an extra dimension
        #i.e., x1 (,d) -> (1,d) flatten array -> 2D array!

    #@profile
    def update_GP(self, x, y):
        #self.change = change
        self.update_obs(x, y)
        #(x,y) -->(xt1,yt1)
        
        ## Can't do internally due to out of memory / invalid array errors from numpy
        self.K_t, self.u, self.phi, f_upd, sigma_upd = update_GP_ext(self.X_t, self.x_t1, self.A, self.y, self.y_t1, self.k_star, self.kvv, self.c, self.kernels)#update_GP_ext(self.X_t, self.x_t1, self.A, self.x_star, self.eta, self.y, self.y_t1, self.k_star, self.variance, self.gamma,self.kvv, self.kernels, self.matern_nu)

        # unsolved ###########why results from two eq below (1)/(2) differ from results from the code , though they are mathmatically identical?  
        #sigtmp = -self.sigma + sigma_upd (1)
        #self.f = self.f + f_upd (2)  

        self.iterate_vars()
        self.f = self.k_star.T @ self.A @ self.y
        self.sigma = self.kvv - self.k_star.T @ self.A @ self.k_star# (V,V)


    #@profile
    def iterate_vars(self):
        self.y = np.append(self.y, self.y_t1)
        self.X_t = np.append(self.X_t, self.x_t1, axis=0)
        self.k_star = np.append(self.k_star, kernel(self.x_t1, self.x_star, self.c, self.kernels), axis=0)  #self.variance, self.gamma, self.kernels, self.matern_nu

        ## update for A eq (27) in note 2.3
        self.A = self.A + self.phi * np.outer(self.u, self.u)
        #print("A shape1",self.A.shape)
        self.A = np.vstack((self.A, -self.phi*self.u.T))
        #print("A shape2",self.A.shape)
        right = np.append(-self.phi*self.u, self.phi)
        #print("A shape3",right.shape)
        self.A = np.column_stack((self.A, right))
        #print("A shape4",self.A.shape)

        self.t += 1

    #@profile
    def max_acq(self):
        test_pt = np.argmax(self.ucb())

        # val = self.f - np.max(self.f) - 1e-4
        # sig = np.diagonal(self.sigma)
        # test_pt = np.argmax(val * norm.cdf(val / sig) + sig * norm.pdf(val))
        if self.test_count[test_pt] >= 3:  # test_counts start with 0, to show max 3 repeated stim, use "\geq"
            test_pt = np.random.choice(np.arange(self.x_star.shape[0]))
            logger.info("Selected stimulus exceeds max tests; Choose randomly ---")
            # arange-generate an array range [0--n-1];np.rd.chocie:sample a number from the array
            #this point has been counted with a largest acq funtion >5 times - local maximum?
        self.test_count[test_pt] += 1
        # logger.info("this this test_pt count (after if) {}".format(self.test_count[test_pt]))
    #    print("test_pt", test_pt)
    #    print("new pt", self.x_star[test_pt])
        return test_pt, self.x_star[test_pt]


    #@profile
    def ucb(self):
        tau = self.d * np.log(self.t + 1e-16) # eq 35
        # import pdb; pdb.set_trace()
        #GP-UCB(x) = µ(x) + (√ντ)σ(x).
        sig = self.sigma
        if np.any(sig < 0):
            sig = np.clip(sig, 0, np.max(sig))
        # if any element in the matrix sig is less than zero
        # np.clip(array, a_min, a_max, out=None)->keep non-neg

        fcn = self.f + np.sqrt(self.nu * tau) * np.sqrt(np.diagonal(sig) + 1e-16)
        #diagonal-sd
        return fcn

    #@profile
    # the original stopping function for single-peak
    def stopping(self):
        # using probability of improvement
        # import pdb; pdb.set_trace()
        val = self.f - np.max(self.f) - 1e-4
        # TODO: error: RuntimeWarning: divide by zero encountered in divide
        sig = np.diagonal(self.sigma) + 1e-12
        PI = np.max(stats.norm.cdf((val) / (sig)))

        # using expected improvement; updated to correct EI calculation
        EI = np.max(val * stats.norm.cdf(val / sig) + sig * stats.norm.pdf(val / sig))
        # combined = (EI + PI)*0.5

        return EI, PI #combined #EI #PI


    #@profile
    def return_par(self):
        return self.sigma,self.f
#@profile
def update_GP_ext(X_t, x_t1, A, y, y_t1, k_star, kvv, c, kernels):#update_GP_ext(X_t, x_t1, A, x_star, eta, y, y_t1, k_star, variance, gamma, kvv, kernels, matern_nu):

    k_t = kernel(X_t, x_t1, c, kernels)#x_t+1
    u = A @ k_t
    k_t1 = kernel(x_t1, x_t1, c, kernels)
    k_star_t1 = kernel(x_t1, c.x_star, c, kernels) #(1,v)
    phi = np.linalg.inv(k_t1 + c.eta**2 - k_t.T.dot(u)) # (1,1)
    # eta float; phi:(1,1); float +- (1,1) or (1,)+-(1,1)-> (1,1) array, e.g., 3 +array[[1]]= array([[4]]); (160,160) -(160,1) = (160,160)
    
    kuk = k_star.T @ u - k_star_t1.T # (V,1) correct x_star (1,1)--> (1,V) => k*t1 (1,V)



    df = np.squeeze(phi * kuk * (((y.T).dot(u)).T - y_t1)) #(1,1) * 2D arr *1D (1,)

    # can't @ here,@ only dim-matached 2D matrix-->2D matrix/1D arr-->1D array;

    dsig = kvv - phi * kuk.dot(kuk.T)#(160, 160)-(160,1) row-wise subtrction
    return k_t, u, phi, df, dsig

# TODO: make this more applicable to other optimizer, maybe eliminate matern_nu?
# or maybe change the input argument to config?
# adapted from pseudo-neurons
def calc_offline_fit(X, y, c, kernels):
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray), "Please convert X and y in numpy array"
    is_X_object = X.dtype == np.object_
    is_y_object = y.dtype == np.object_
    is_y_multi = (y.ndim > 1) or (is_y_object)
    print("X is object?", is_X_object)
    print("y is object?", is_y_object)
    # print("y contains multiple neurons?", is_y_multi)
    # Detect single vs multiple neuron case
    if not is_y_multi:
        print("Detected single neuron case")
        neuron_len = 1
    else:
        print("Detected multiple neuron case")
        neuron_len = y.shape[0]
        print(neuron_len)
    f = np.empty((neuron_len, c.x_star.shape[0])) # the number of neurons
    sigma = np.empty((neuron_len, c.x_star.shape[0], c.x_star.shape[0]))
    if not is_X_object:
        if not is_y_multi:
            assert X.shape[0] == y.shape[0] and X.shape[1] == c.d, \
                f"Expected X.shape = (T, d) and y.shape = (T,), got {X.shape} and {y.shape}"
        else:
            assert X.shape[0] == y.shape[1] and X.shape[1] == c.d, \
                f"Expected X.shape = (T, d) and y.shape = (n, T), got {X.shape} and {y.shape}"

    for neuron in range(neuron_len):  #y.shape[0]):
        print(neuron)
        if not is_X_object:  # many neurons, but X is homogeneous
            X_t = X
            y_neuron = y if not is_y_multi else y[neuron]
            # X_t = X[neuron]  # added "[neuron]" for X's that have different length for different neurons
            # y_neuron = y[neuron] #1D arr (T,)
        else:  # X is inhomogeneous
            X_t = X[neuron]
            y_neuron = y[neuron]
        # Kernel matrices
        T = X_t.shape[0]
        # print(is_multiple, type(X_t), X_t)
        K_t = kernel(X_t, X_t, c, kernels)

        k_star = kernel(X_t, c.x_star, c, kernels)
        kvv = kernel(c.x_star, c.x_star, c, kernels)

        # Compute GP parameters
        A = np.linalg.inv(K_t + c.eta**2 * np.eye(T))     # TO DO - look at pinv 
        f[neuron] = k_star.T @ A @ y_neuron                    # predicted means
        sigma[neuron] = c.var * np.eye(c.x_star.shape[0]) - k_star.T @ A @ k_star    # covariance matrix
        t = T

    return f, sigma

# # online fits
# TODO: we should change it to single neuron basis
def calc_online_fit(X, y, c, kernels, init_T, end_T):
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray), "Please convert X and y in numpy array"
    is_X_object = X.dtype == np.object_
    is_y_object = y.dtype == np.object_
    is_y_multi = (y.ndim > 1) or (is_y_object)
    print("X is object?", is_X_object)
    print("y is object?", is_y_object)
    # print("y contains multiple neurons?", is_y_multi)
    # Detect single vs multiple neuron case
    if not is_y_multi:
        print("Detected single neuron case")
        neuron_len = 1
    else:
        print("Detected multiple neuron case")
        neuron_len = y.shape[0]
        print(neuron_len)
    online_len = end_T - init_T if end_T > 0 else X.shape[0] + end_T + 1 - init_T
    print("online fit length:", online_len)
    # f = np.empty((neuron_len, c.x_star.shape[0])) # the number of neurons
    # sigma = np.empty((neuron_len, c.x_star.shape[0], c.x_star.shape[0]))
    f = np.empty((online_len, c.x_star.shape[0]))
    sigma = np.empty((online_len, c.x_star.shape[0], c.x_star.shape[0]))
    stopping_crits = np.empty((online_len, 2))  # getting both EI and PI i assume 
    peak_lc = np.empty((online_len, 5))
    for neuron in range(neuron_len):  # number of neurons
        if is_y_multi:
            print(f"N: {neuron}")
        # X_neuron = X[neuron]
        # y_neuron = y[neuron]
        if not is_X_object:  # many neurons, but X is homogeneous
            X_neuron = X
            y_neuron = y if not is_y_multi else y[neuron]
            # X_t = X[neuron]  # added "[neuron]" for X's that have different length for different neurons
            # y_neuron = y[neuron] #1D arr (T,)
        else:  # X is inhomogeneous
            X_neuron = X[neuron]
            y_neuron = y[neuron]
        end_T_neuron = X_neuron.shape[0] + end_T + 1 if end_T < 0 else end_T
        init_T_neuron = end_T_neuron if init_T > end_T_neuron else init_T
        assert end_T_neuron <= X_neuron.shape[0] and end_T_neuron >= init_T_neuron, "Houston we have a problem"
        optimizer = Optimizer(c=c, kernels=kernels)
        initial_X = X_neuron[:init_T_neuron] # Shape: (1, 2)
        initial_y = np.array(y_neuron[0:init_T_neuron])  #[0:4]
        optimizer.initialize_GP(X=initial_X, y=initial_y)
        print(f"from {init_T_neuron} to {end_T_neuron}")
        counter = 0
        for idx in range(init_T_neuron, end_T_neuron):  # df_test.shape[0]
            incom_X = X_neuron[idx]
            incom_y = np.array(y_neuron[idx])
            optimizer.update_GP(x=incom_X, y=incom_y)
            EI, PI = optimizer.stopping()

            max_idx = np.argmax(optimizer.f)
            peak_lc[counter] = tuple(list(map(int, c.x_star[max_idx])))
            f[counter] = optimizer.f
            sigma[counter] = optimizer.sigma
            stopping_crits[counter] = [EI, PI]
            
            counter += 1
        print(counter)
    return f, sigma, stopping_crits, peak_lc

# TODO: should the follow two functions be here or in util.py?
def find_peaks(f_array, x_star):
    return x_star[np.argmax(f_array, axis = 1)]

# calculate the mean squared error between matrices (of_peak = False) or peak locations (of_peak = True)
# run offline. aka after getting the offline matrices
def mse_on_offline(f_online, f_offline, x_star, of_peak = True):
    assert f_online.shape == f_offline.shape, "The dim doesn't match"
    # mse_on_offline_ls = np.zeros((f_online.shape[0], ))  # (146, ), as MSE is just one number
    if of_peak:  # examining the PEAK LOCATION of the tuning curves
        peaks_online = find_peaks(f_online, x_star)
        peaks_offline = find_peaks(f_offline, x_star)
        assert peaks_online.shape == peaks_offline.shape, "The dim of peaks array doesn't match"
        mse_on_offline_ls = np.array([
            mean_squared_error(peaks_online[i], peaks_offline[i]) 
            for i in range(f_online.shape[0])])
    else: # calculate MSE across all stimulus-response locations
        mse_on_offline_ls = np.array([
            mean_squared_error(f_online[i], f_offline[i]) 
            for i in range(f_online.shape[0])])
    return mse_on_offline_ls


import numpy as np
import numba as nb
import scipy.special

# @profile
# @nb.njit(fastmath=True,parallel=True)
def kernel_rbf_old(x, x_j, var, gamma): #dim: (A,d) , (B,d) --> (A,B)
    K_rbf=np.dot(gamma*x,x_j.T)

    TMP_x=np.empty(x.shape[0],dtype=x.dtype)
    for i in nb.prange(x.shape[0]):
         TMP_x[i]=np.dot(gamma,(x[i]**2))

    TMP_x_j=np.empty(x_j.shape[0],dtype=x_j.dtype)
    for i in nb.prange(x_j.shape[0]):
        TMP_x_j[i]=np.dot(gamma,(x_j[i]**2))

    for i in nb.prange(x.shape[0]):
        for j in range(x_j.shape[0]):
            K_rbf[i,j]=var*np.exp(-(-2.0*K_rbf[i,j]+TMP_x[i]+TMP_x_j[j]))

    return K_rbf

# this is more like a discrete switch --> reach something then teleport to the original (?)
# legacy code, kept for reference
def kernel_rbf_periodic_old(x, x_j, gamma):  #var, gamma
    K_rbf_periodic = np.zeros((x.shape[0], x_j.shape[0]))

    for i in range(x.shape[0]):
        for j in range(x_j.shape[0]):
            # dist = np.abs(x[i,0] - x_j[j,0])
            dist = np.abs(x[i] - x_j[j])  # 1d implementation, for now
            if dist > 10:
                dist = 10 - dist
            K_rbf_periodic[i,j] = np.exp(-gamma * ((dist)**2))  # changed gamma[0] to gamma
    
    return K_rbf_periodic

# ---- per-dimension helpers (all vectorized) ----
def kernel_rbf(x, x_j, gamma):
    """RBF for a single dimension: exp(-gamma * (x - x')^2)."""
    dist = x[:, None] - x_j[None, :]
    return np.exp(-gamma * dist**2)

def kernel_rbf_periodic(x, x_j, gamma, period):
    """Periodic RBF for a single dimension with period P"""
    dist = np.abs(x[:, None] - x_j[None, :])
    p = float(period)
    ls = 0.5 / float(gamma)
    return np.exp(-2 * np.sin((dist * np.pi) / p)**2 / (ls**2)) * np.exp(-gamma * dist**2)

def kernel_linear_dim(x, x_j):
    """Linear kernel for one dim: outer product."""
    return np.outer(x, x_j)

def kernel_linear_multi(X_lin, Xj_lin):
    """Linear kernel for multiple dims at once: X X'^T."""
    return X_lin @ Xj_lin.T

def kernel_matern(x, x_j, gamma, matern_nu):
    """Matern-nu for one dim with lengthscale ls = 0.5/gamma."""
    dist = np.abs(x[:, None] - x_j[None, :])
    ls = 0.5 / float(gamma)

    if matern_nu == 0.5:
        return np.exp(-dist / ls)
    if matern_nu == 1.5:
        t = np.sqrt(3.0) * dist / ls
        return (1.0 + t) * np.exp(-t)
    if matern_nu == 2.5:
        t = np.sqrt(5.0) * dist / ls
        return (1.0 + t + (t**2) / 3.0) * np.exp(-t)

    # General matern_nu (slower due to Bessel Kν)
    t = np.sqrt(2.0 * matern_nu) * dist / ls
    t = np.where(t == 0, np.finfo(float).eps, t)
    coef = 2.0**(1.0 - matern_nu) / scipy.special.gamma(matern_nu)
    return coef * (t**matern_nu) * scipy.special.kv(matern_nu, t)


# the main kernel function
def kernel(x, x_j, c, kernels):
    K = np.ones((x.shape[0], x_j.shape[0]), dtype=np.float64) 
    gamma = np.asarray(c.gamma, dtype=float)

    # handle linear dims first so multiple linear dims are fused
    # TODO: check if this implementation is correct
    idx_linear = [i for i, k in enumerate(kernels) if k == 'linear']
    if idx_linear:
        if len(idx_linear) == 1:
            d = idx_linear[0]
            K *= kernel_linear_dim(x[:, d], x_j[:, d])
        else:
            K *= kernel_linear_multi(x[:, idx_linear], x_j[:, idx_linear])

    # other dims: call the respective helper inside the if/elif
    used_stationary = False  # track if we used rbf/matern/periodic
    for i, ki in enumerate(kernels):
        if ki == 'linear':
            continue  # already handled
        elif ki == 'rbf':
            K *= kernel_rbf(x[:, i], x_j[:, i], gamma[i])
            used_stationary = True
        elif ki == 'rbf_periodic':
            period = getattr(c, "periodic_p", None)
            if period is None:
                raise ValueError("Periodic kernel requested but c.period is missing.")
            P_i = period if np.ndim(period) == 0 else period[i]
            K *= kernel_rbf_periodic(x[:, i], x_j[:, i], gamma[i], P_i)
            used_stationary = True
        elif ki == 'matern':
            nu = getattr(c, "matern_nu", None)
            if nu is None:
                raise ValueError("Matern kernel requested but c.matern_nu is missing.")
            nu_i = float(nu) if np.ndim(nu) == 0 else float(nu[i])
            K *= kernel_matern(x[:, i], x_j[:, i], gamma[i], nu_i)
            used_stationary = True
        else:
            raise ValueError(f"Unknown kernel '{ki}' at dim {i}")

    if used_stationary:
        K *= float(c.var)

    return K

# # this is the code we used in improv
# # def kernel(x, x_j, variance, gamma):

# #     # new ways to compute kernel
# #     dist = x[:, None, :] - x_j[None, :, :]
# #     ws_dist = np.sum(gamma * (dist**2), axis =2)
# #     K = variance *np.exp(-ws_dist)
            
# #     return K

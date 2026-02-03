import numba as nb
import scipy.special

# benchmarking
from line_profiler import profile
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

# @profile
# @nb.njit(fastmath=True,parallel=True)
def kernel_rbf_old(x, x_j, var, gamma): #dim: (A,d) , (B,d) --> (A,B)
    K_rbf=jnp.dot(gamma*x,x_j.T)

    TMP_x=jnp.empty(x.shape[0],dtype=x.dtype)
    for i in nb.prange(x.shape[0]):
         TMP_x[i]=jnp.dot(gamma,(x[i]**2))

    TMP_x_j=jnp.empty(x_j.shape[0],dtype=x_j.dtype)
    for i in nb.prange(x_j.shape[0]):
        TMP_x_j[i]=jnp.dot(gamma,(x_j[i]**2))

    for i in nb.prange(x.shape[0]):
        for j in range(x_j.shape[0]):
            K_rbf[i,j]=var*jnp.exp(-(-2.0*K_rbf[i,j]+TMP_x[i]+TMP_x_j[j]))

    return K_rbf

# this is more like a discrete switch --> reach something then teleport to the original (?)
# legacy code, kept for reference
def kernel_rbf_periodic_old(x, x_j, gamma):  #var, gamma
    K_rbf_periodic = jnp.zeros((x.shape[0], x_j.shape[0]))

    for i in range(x.shape[0]):
        for j in range(x_j.shape[0]):
            # dist = jnp.abs(x[i,0] - x_j[j,0])
            dist = jnp.abs(x[i] - x_j[j])  # 1d implementation, for now
            if dist > 10:
                dist = 10 - dist
            K_rbf_periodic.at[i,j].set(jnp.exp(-gamma * ((dist)**2)))  # changed gamma[0] to gamma
    
    return K_rbf_periodic

# ---- per-dimension helpers (all vectorized) ----
def kernel_rbf(x, x_j, gamma):
    """RBF for a single dimension: exp(-gamma * (x - x')^2)."""
    dist = x[:, None] - x_j[None, :]
    return jnp.exp(-gamma * dist**2)

def kernel_rbf_periodic(x, x_j, gamma, period):
    """Periodic RBF for a single dimension with period P"""
    dist = jnp.abs(x[:, None] - x_j[None, :])
    p = float(period)
    ls = 0.5 / float(gamma)
    return jnp.exp(-2 * jnp.sin((dist * jnp.pi) / p)**2 / (ls**2)) * jnp.exp(-gamma * dist**2)

def kernel_linear_dim(x, x_j):
    """Linear kernel for one dim: outer product."""
    return jnp.outer(x, x_j)

def kernel_linear_multi(X_lin, Xj_lin):
    """Linear kernel for multiple dims at once: X X'^T."""
    return X_lin @ Xj_lin.T

def kernel_matern(x, x_j, gamma, matern_nu):
    """Matern-nu for one dim with lengthscale ls = 0.5/gamma."""
    dist = jnp.abs(x[:, None] - x_j[None, :])
    ls = 0.5 / float(gamma)

    if matern_nu == 0.5:
        return jnp.exp(-dist / ls)
    if matern_nu == 1.5:
        t = jnp.sqrt(3.0) * dist / ls
        return (1.0 + t) * jnp.exp(-t)
    if matern_nu == 2.5:
        t = jnp.sqrt(5.0) * dist / ls
        return (1.0 + t + (t**2) / 3.0) * jnp.exp(-t)

    # General matern_nu (slower due to Bessel Kν)
    t = jnp.sqrt(2.0 * matern_nu) * dist / ls
    t = jnp.where(t == 0, jnp.finfo(float).eps, t)
    coef = 2.0**(1.0 - matern_nu) / scipy.special.gamma(matern_nu)
    return coef * (t**matern_nu) * scipy.special.kv(matern_nu, t)


# the main kernel function
@partial(jax.jit, static_argnames=['c', 'kernels'])
# @profile # needs to go after partial or it doesn't work
def kernel(x, x_j, c, kernels):
    K = jnp.ones((x.shape[0], x_j.shape[0])) 
    gamma = jnp.asarray(c.gamma, dtype=float)

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
            P_i = period if jnp.ndim(period) == 0 else period[i]
            K *= kernel_rbf_periodic(x[:, i], x_j[:, i], gamma[i], P_i)
            used_stationary = True
        elif ki == 'matern':
            nu = getattr(c, "matern_nu", None)
            if nu is None:
                raise ValueError("Matern kernel requested but c.matern_nu is missing.")
            nu_i = float(nu) if jnp.ndim(nu) == 0 else float(nu[i])
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
# #     ws_dist = jnp.sum(gamma * (dist**2), axis =2)
# #     K = variance *jnp.exp(-ws_dist)
            
# #     return K

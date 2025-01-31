import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as np

from . import classy_sz  # shared instance
from .profiles import y_ell_interpolate
from .massfuncs import get_hmf_at_z_and_m

def simpson(y, *, x=None, dx=1.0, axis=-1):
    y = jnp.asarray(y)
    nd = len(y.shape)
    N = y.shape[axis]
    last_dx = dx
    if x is not None:
        x = jnp.asarray(x)
        if len(x.shape) == 1:
            shapex = [1] * nd
            shapex[axis] = x.shape[0]
            saveshape = x.shape
            returnshape = 1
            x = x.reshape(tuple(shapex))
        elif len(x.shape) != len(y.shape):
            raise ValueError("If given, shape of x must be 1-D or the "
                             "same as y.")
        if x.shape[axis] != N:
            raise ValueError("If given, length of x along axis must be the "
                             "same as y.")
    if N % 2 == 0:
        val = 0.0
        result = 0.0
        slice_all = (slice(None),) * nd
        if N == 2:
            # need at least 3 points in integration axis to form parabolic
            # segment. If there are two points then any of 'avg', 'first',
            # 'last' should give the same result.
            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            if x is not None:
                last_dx = x[slice1] - x[slice2]
            val += 0.5 * last_dx * (y[slice1] + y[slice2])
        else:
            # use Simpson's rule on first intervals
            result = _basic_simpson(y, 0, N-3, x, dx, axis)
            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            slice3 = tupleset(slice_all, axis, -3)
            h = jnp.asarray([dx, dx], dtype=jnp.float64)
            if x is not None:
                # grab the last two spacings from the appropriate axis
                hm2 = tupleset(slice_all, axis, slice(-2, -1, 1))
                hm1 = tupleset(slice_all, axis, slice(-1, None, 1))
                diffs = jnp.float64(jnp.diff(x, axis=axis))
                h = [jnp.squeeze(diffs[hm2], axis=axis),
                     jnp.squeeze(diffs[hm1], axis=axis)]
            num = 2 * h[1] ** 2 + 3 * h[0] * h[1]
            den = 6 * (h[1] + h[0])
            alpha = jnp.true_divide(
                num,
                den,
            )
            num = h[1] ** 2 + 3.0 * h[0] * h[1]
            den = 6 * h[0]
            beta = jnp.true_divide(
                num,
                den,
            )
            num = 1 * h[1] ** 3
            den = 6 * h[0] * (h[0] + h[1])
            eta = jnp.true_divide(
                num,
                den,
            )
            result += alpha*y[slice1] + beta*y[slice2] - eta*y[slice3]
        result += val
    else:
        result = _basic_simpson(y, 0, N-2, x, dx, axis)
    if returnshape:
        x = x.reshape(saveshape)
    return result
def _basic_simpson(y, start, stop, x, dx, axis):
    nd = len(y.shape)
    if start is None:
        start = 0
    step = 2
    slice_all = (slice(None),)*nd
    slice0 = tupleset(slice_all, axis, slice(start, stop, step))
    slice1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
    slice2 = tupleset(slice_all, axis, slice(start+2, stop+2, step))
    if x is None:  # Even-spaced Simpson's rule.
        result = jnp.sum(y[slice0] + 4.0*y[slice1] + y[slice2], axis=axis)
        result *= dx / 3.0
    else:
        # Account for possibly different spacings.
        #    Simpson's rule changes a bit.
        h = jnp.diff(x, axis=axis)
        sl0 = tupleset(slice_all, axis, slice(start, stop, step))
        sl1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
        h0 = h[sl0].astype(float)
        h1 = h[sl1].astype(float)
        hsum = h0 + h1
        hprod = h0 * h1
        h0divh1 = jnp.true_divide(h0, h1)
        tmp = hsum/6.0 * (y[slice0] *
                          (2.0 - jnp.true_divide(1.0, h0divh1)) +
                          y[slice1] * (hsum *
                                       jnp.true_divide(hsum, hprod)) +
                          y[slice2] * (2.0 - h0divh1))
        result = jnp.sum(tmp, axis=axis)
    return result
def tupleset(t, i, value):
    l = list(t)
    l[i] = value
    return tuple(l)

def dVdzdOmega(z, params_values_dict = None):

    rparams = classy_sz.get_all_relevant_params(params_values_dict = params_values_dict)
    h = rparams['h']
    dAz = classy_sz.get_angular_distance_at_z(z,params_values_dict = params_values_dict) * h
    # dAz = classy_sz.get_angular_distance_at_z(z,params_values_dict = cosmo_params)/(1+z)*h # in Mpc/h
    Hz = classy_sz.get_hubble_at_z(z,params_values_dict = params_values_dict) / h # in Mpc^(-1) h
    # print(Hz)

    return (1+z)**2*dAz**2/Hz

def get_ell_range(params_values_dict = None):

    # Get cosmological parameters
    rparams = classy_sz.get_all_relevant_params(params_values_dict = params_values_dict)
    l_min = rparams['ell_min']
    l_max = rparams['ell_max']
    dlogell = rparams['dlogell']

    log10_l_min = jnp.log10(l_min)
    log10_l_max = jnp.log10(l_max)
    num = int((log10_l_max - log10_l_min) / dlogell) + 1
    # print(num)

    # Define evaluation ell values (uniform grid)
    ell_eval = jnp.logspace(log10_l_min, log10_l_max, num=num)
    
    return ell_eval

def get_integral_grid(params_values_dict = None):
    
    rparams = classy_sz.get_all_relevant_params(params_values_dict = params_values_dict)
    allparams = classy_sz.pars
    

    z_min = allparams['z_min']
    z_max = allparams['z_max']
    z_grid = jnp.geomspace(z_min, z_max, 100)
    
    h = rparams['h']
    # Define an m_grid:
    M_min = allparams['M_min']
    M_max = allparams['M_max']
    m_grid_yl = jnp.geomspace(M_min,M_max,100)
    # m_grid_dndlnm = jnp.geomspace(M_min,M_max,100) * h


    def get_yellm_for_z(zp):
        ell, y_ellm = y_ell_interpolate(zp, m_grid_yl, params_values_dict = params_values_dict)
        # print(y_ellm)
        # print(ell.shape)
        # ell, y_ellm = y_ell_complete(zp, m_grid_yl, params_values_dict=cosmo_params)
        return ell, y_ellm
    
    def get_hmf_for_z(zp):
        # dndlnm = get_hmf_at_z_and_m(z = zp, m=m_grid_dndlnm, params_values_dict=cosmo_params)
        dndlnm = get_hmf_at_z_and_m(z = zp, m=m_grid_yl, params_values_dict= params_values_dict)
        return dndlnm
 

    # Vectorize this function over `z_grid`
    # y_ell_mz_grid = jax.vmap(get_yellm_for_z)(z_grid)
    ell, y_ell_mz_grid = jax.vmap(get_yellm_for_z)(z_grid)
    dndlnm_grid = jax.vmap(get_hmf_for_z)(z_grid)
    # print(y_ell_mz_grid.shape)
    # print(dndlnm_grid.shape)

    # print(z_grid)
    # print(y_ell_mz_grid)
    # Ensure `dndlnm_grid` has a compatible shape for broadcasting
    dndlnm_grid_expanded = dndlnm_grid[:, :, None]  # Shape becomes (100, 100, 1)

    comov_vol = dVdzdOmega(z_grid, params_values_dict=params_values_dict)

    # Expand comov_vol to align with the shape of `result`
    comov_vol_expanded = comov_vol[:, None, None]  # Shape becomes (100, 1, 1)

    # Perform element-wise multiplication
    result = y_ell_mz_grid**2 * dndlnm_grid_expanded * comov_vol_expanded  # Shape becomes (100, 100, 18)= (dim_z, dim_m, dim_ell)
    # result = y_ell_mz_grid
    # prefactor = y_ell_prefactor(z_grid, m_grid_yl, params_values_dict=cosmo_params)

    # Perform element-wise multiplication
    # result = y_ell_mz_grid * dndlnm_grid_expanded  # Shape becomes (100, 100, 18)  
    # print(result.shape) 
    # print(ell.shape)
    return result
    

def compute_integral(params_values_dict = None):

    allparams = classy_sz.pars

    integrand = get_integral_grid(params_values_dict = params_values_dict) # shape is (dim_z, dim_m, dim_ell) 
    # ell = get_integral_grid()[0]
    # print(integrand.shape)
    # print(ell.shape)

    z_min = allparams['z_min']
    z_max = allparams['z_max']
    z_grid = jnp.geomspace(z_min, z_max, 100)
    

    # Define an m_grid:
    M_min = allparams['M_min']
    M_max = allparams['M_max']
    m_grid = jnp.geomspace(M_min,M_max,100)
    logm_grid = jnp.log(m_grid)


    ell = get_ell_range()
    # ell = y_ell_complete(z=1, m=m_grid, params_values_dict = cosmo_params)[0]
    # This will store the integrated value for each ell
    C_yy = jnp.zeros(len(ell))

    for i in range(len(ell)):
        # 1) Integrate over m
        #    integrand[:, :, i] has shape (dim_z, dim_m)
        partial_m = simpson(integrand[:, :, i], x=logm_grid, dx=(logm_grid[1]-logm_grid[0]),axis=1)
        # partial_m = jnp.trapezoid(integrand[:, :, i], x=logm_grid, dx=(logm_grid[1]-logm_grid[0]),axis=1)
        # partial_m = simpson(integrand[:, :, i], x=m_grid, axis=1)
        # partial_m now has shape (dim_z,)

        # 2) Integrate the result over z
        result = simpson(partial_m, x=z_grid, dx = (z_grid[1]-z_grid[0]), axis=0)
        # result = jnp.trapezoid(partial_m, x=z_grid, dx = (z_grid[1]-z_grid[0]), axis=0)

        # Store the result for this ell
        C_yy = C_yy.at[i].set(result)
            
    return C_yy  

def get_integral_grid_trisp(params_values_dict=None):

    # 1) Get y_\ell(z, m) over grids of z and m
    rparams = classy_sz.get_all_relevant_params(params_values_dict=params_values_dict)
    allparams = classy_sz.pars
    
    z_min = allparams['z_min']
    z_max = allparams['z_max']
    z_grid = jnp.geomspace(z_min, z_max, 100)

    M_min = allparams['M_min']
    M_max = allparams['M_max']
    m_grid = jnp.geomspace(M_min, M_max, 100)

    # y_ell_mz_grid: shape = (n_z, n_m, n_ell)
    def get_yellm_for_z(zp):
        # Returns ell (length n_ell) and y_ellm (length n_m x n_ell)
        # but typically we stack in shape (n_m, n_ell)
        ell_vals, y_ellm = y_ell_interpolate(zp, m_grid, params_values_dict=params_values_dict)
        return y_ellm
    

    # Vectorize over z
    y_ell_mz_grid = jax.vmap(get_yellm_for_z)(z_grid)  
    # shape = (n_z, n_m, n_ell)

    # Save ell from a single call (assuming same for all z):
    ell_vals, _ = y_ell_interpolate(z_grid[0], m_grid, params_values_dict=params_values_dict)
    # or keep a separate function get_ell_range() if you prefer
    ell = ell_vals  # shape = (n_ell,)

    # 2) Get halo mass function dndlnm over the same z,m
    def get_hmf_for_z(zp):
        return get_hmf_at_z_and_m(z=zp, m=m_grid, params_values_dict=params_values_dict)
    dndlnm_grid = jax.vmap(get_hmf_for_z)(z_grid)  
    # shape = (n_z, n_m)

    # 3) Get comoving volume element dV/dz/dOmega over z
    comov_vol = dVdzdOmega(z_grid, params_values_dict=params_values_dict)
    # shape = (n_z,)

    # Expand dimensions for consistent broadcasting:
    #   dndlnm_grid_expanded: shape (n_z, n_m, 1, 1)
    dndlnm_grid_expanded = dndlnm_grid[:, :, None, None]
    #   comov_vol_expanded: shape (n_z, 1, 1, 1)
    comov_vol_expanded   = comov_vol[:, None, None, None]

    # 4) Construct the integrand:
    # y^2 for each ell
    y_ell_sq = y_ell_mz_grid**2  # shape: (n_z, n_m, n_ell)

    # We need (n_z, n_m, n_ell, n_ell'). 
    # The product y_ell_sq[..., None] * y_ell_sq[..., None, :] 
    # gives shape (n_z, n_m, n_ell, n_ell)
    # i.e. y_ell^2 * y_ell'^2
    integrand = y_ell_sq[:, :, :, None] * y_ell_sq[:, :, None, :]
    # print(integrand.shape)

    # Now multiply by dndlnm and comov. 
    integrand = integrand * dndlnm_grid_expanded * comov_vol_expanded
    # final shape = (n_z, n_m, n_ell, n_ell)

    return ell, integrand


def compute_trispectrum(params_values_dict=None):
    # 1) Build integrand
    ell, integrand = get_integral_grid_trisp(params_values_dict)

    # 2) Construct z and m grids (consistent with what's in get_integral_grid_trisp)
    allparams = classy_sz.pars
    z_min = allparams['z_min']
    z_max = allparams['z_max']
    z_grid = jnp.geomspace(z_min, z_max, 100)
    
    M_min = allparams['M_min']
    M_max = allparams['M_max']
    m_grid = jnp.geomspace(M_min, M_max, 100)
    logm_grid = jnp.log(m_grid)

    # integrand shape = (n_z, n_m, n_ell, n_ell')

    # 3) Integrate over m (axis=1) using log(m) or m—depending on your definition
    # partial_m shape = (n_z, n_ell, n_ell')
    partial_m = simpson(integrand, x=logm_grid, axis=1)  

    # 4) Integrate over z (axis=0)
    # final shape = (n_ell, n_ell')
    T_ell_ellprime = simpson(partial_m, x=z_grid, axis=0)

    # T_ell_ellprime[i,j] ~ T_{ell_i, ell_j}
    return ell, T_ell_ellprime


def compute_tsz_covariance(params_values_dict=None, noise_ell=None, f_sky=1.0):
    """
    Returns M_llp
    Time to compute this = time to compute tSZ power spectrum + time to compute tSZ trispectrum
    """

    # 1) Compute the tSZ power spectrum C_ell^{yy}
    C_yy = compute_integral(params_values_dict=params_values_dict)  
    # Shape: (n_ell,)

    # 2) Compute the tSZ trispectrum T_{ell,ell'}^{yy} and grab the ell array
    ell_arr, T_ell_ellprime = compute_trispectrum(params_values_dict=params_values_dict)
    # T_ell_ellprime shape: (n_ell, n_ell)
    # ell_arr shape:        (n_ell,)

    # 3) If no noise is given, set it to zero
    if noise_ell is None:
        noise_ell = jnp.zeros_like(C_yy)
    # noise_ell shape: (n_ell,)

    # 4) Construct the diagonal term
    #    diag_term[ell] = [4π (C_ell + N_ell)^2] / [ell + 1/2]
    diag_term = (4.0 * jnp.pi) * (C_yy + noise_ell)**2 / (ell_arr + 0.5)

    # 5) Build the full covariance matrix
    #    M = diag_term * δ_{ell,ell'} + T_{ell,ell'}
    #    Then multiply by 1 / [4π f_sky]
    M = jnp.diag(diag_term) + T_ell_ellprime

    M_G = jnp.diag(diag_term)/ (4.0 * jnp.pi * f_sky)

    M = M / (4.0 * jnp.pi * f_sky)

    return ell_arr, M, M_G


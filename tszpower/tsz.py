import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as np
from . import classy_sz  # shared instance
from .profiles import y_ell_interpolate
from .massfuncs import get_hmf_at_z_and_m
from .utils import get_ell_range, simpson
import time
from jax import lax

def dVdzdOmega(z, params_values_dict = None):

    rparams = classy_sz.get_all_relevant_params(params_values_dict = params_values_dict)
    h = rparams['h']
    dAz = classy_sz.get_angular_distance_at_z(z,params_values_dict = params_values_dict) * h
    # dAz = classy_sz.get_angular_distance_at_z(z,params_values_dict = cosmo_params)/(1+z)*h # in Mpc/h
    Hz = classy_sz.get_hubble_at_z(z,params_values_dict = params_values_dict) / h # in Mpc^(-1) h
    # print(Hz)

    return (1+z)**2*dAz**2/Hz


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

@jax.jit
def compute_integral(params_values_dict = None):

    allparams = classy_sz.pars
    # print("PASS")
    # start_time = time.time()
    integrand = get_integral_grid(params_values_dict = params_values_dict) # shape is (dim_z, dim_m, dim_ell) 
    # integrand = integrand_grid
    # ell = get_integral_grid()[0]
    # print(integrand.shape)
    # print(ell.shape)
    # print("THis is passed")

    z_min = allparams['z_min']
    z_max = allparams['z_max']
    z_grid = jnp.geomspace(z_min, z_max, 100)
    

    # Define an m_grid:
    M_min = allparams['M_min']
    M_max = allparams['M_max']
    m_grid = jnp.geomspace(M_min,M_max,100)
    logm_grid = jnp.log(m_grid)

    # Calculate the spacings (assumed uniform in log-space and z)
    dx_m = logm_grid[1] - logm_grid[0]
    dx_z = z_grid[1] - z_grid[0]

    # Get the ell array (should be a JAX array)
    ell = get_ell_range()  # shape (n_ell,)
    n_ell = ell.shape[0]
    # ell = y_ell_complete(z=1, m=m_grid, params_values_dict = cosmo_params)[0]
    # This will store the integrated value for each ell
    C_yy = jnp.zeros(len(ell))
    # end_time = time.time()
    # print(f"intermediate 1 took {end_time - start_time:.4f} seconds")
    # start_time = time.time()
    # for i in range(len(ell)):
    #     # 1) Integrate over m
    #     #    integrand[:, :, i] has shape (dim_z, dim_m)
    #     partial_m = simpson(integrand[:, :, i], x=logm_grid, dx=(logm_grid[1]-logm_grid[0]),axis=1)
    #     # partial_m = jnp.trapezoid(integrand[:, :, i], x=logm_grid, dx=(logm_grid[1]-logm_grid[0]),axis=1)
    #     # partial_m = simpson(integrand[:, :, i], x=m_grid, axis=1)
    #     # partial_m now has shape (dim_z,)

    #     # 2) Integrate the result over z
    #     result = simpson(partial_m, x=z_grid, dx = (z_grid[1]-z_grid[0]), axis=0)
    #     # result = jnp.trapezoid(partial_m, x=z_grid, dx = (z_grid[1]-z_grid[0]), axis=0)

    #     # Store the result for this ell
    #     C_yy = C_yy.at[i].set(result)
    # end_time = time.time()
    # print(f"intermediate 2 took {end_time - start_time:.4f} seconds")

    # Define a scan body function that, for a given index i, computes the integrated value.
    def scan_body(_, i):
        # For the i-th ell value, extract the corresponding slice of the integrand:
        #   integrand_i has shape (n_z, n_m)
        integrand_i = integrand[:, :, i]
        
        # First integrate over m (using your Simpson routine along axis=1)
        partial_m = simpson(integrand_i, x=logm_grid, dx=dx_m, axis=1)  # shape (n_z,)
        
        # Then integrate the result over z (along axis=0)
        result = simpson(partial_m, x=z_grid, dx=dx_z, axis=0)  # scalar
        
        return None, result

    # Use lax.scan over the indices 0,1,..., n_ell-1.
    # The carry value is not used here (set to None).
    _, C_yy = lax.scan(scan_body, None, jnp.arange(n_ell))
    # C_yy is an array of shape (n_ell,)
    # print("PASS COMPUTE INTEGRAL")
            
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
    # partial_m = jnp.trapezoid(integrand, x=logm_grid, axis=1)

    # 4) Integrate over z (axis=0)
    # final shape = (n_ell, n_ell')
    T_ell_ellprime = simpson(partial_m, x=z_grid, axis=0)
    # T_ell_ellprime = jnp.trapezoid(partial_m, x=z_grid, axis=0)

    # T_ell_ellprime[i,j] ~ T_{ell_i, ell_j}
    return ell, T_ell_ellprime


def compute_tsz_covariance(params_values_dict=None, noise_ell=None, f_sky=1.0):
    """
    Computes the binned tSZ covariance evaluated at the pre-defined ell values.
    
    The covariance is given by:
    
      M_ll' =  2*(C_l + N_l)^2 / [(2l+1)*Delta_ell * f_sky] * delta_ll'  +  T_ll'/(4pi f_sky)
    
    where:
      - ell (bin centers) are defined in get_ell_range().
      - Delta_ell is the width of the multipole bin (computed from the bin centers).
      - T_ll' is the tSZ trispectrum.
    """
    # 1) Use the pre-defined ell centers.
    ell_arr = get_ell_range()  # shape: (n_bins,)
    
    # 2) Compute the tSZ power spectrum at these ell values.
    #    (Assume compute_integral returns an array of shape (n_bins,))
    C_yy = compute_integral(params_values_dict=params_values_dict)
    
    # 3) Compute the tSZ trispectrum at these ell values.
    #    (Assume compute_trispectrum returns both the ell array and T with shape (n_bins, n_bins))
    ell_arr_T, T_ell_ellprime = compute_trispectrum(params_values_dict=params_values_dict)
    # (You may wish to check that ell_arr_T is consistent with ell_arr.)
    
    # 4) Set the noise if not provided.
    if noise_ell is None:
        noise_ell = jnp.zeros_like(C_yy)
    
    # 5) Compute the bin widths (Delta_ell) from the bin centers.
    #     Assume bin edges lie halfway between the centers.
    #     For a given ell_arr = [ell0, ell1, ..., ell_{n-1}], define:
    #         edge0 = ell0 - 0.5*(ell1 - ell0)
    #         edge_i = 0.5*(ell_{i-1}+ell_i)   for i=1,...,n-1
    #         edge_n = ell_{n-1} + 0.5*(ell_{n-1}-ell_{n-2})
    #     Then Delta_ell[i] = edge_{i+1} - edge_i.
    n_bins = ell_arr.shape[0]
    edges = jnp.concatenate((
        jnp.array([ell_arr[0] - 0.5*(ell_arr[1]-ell_arr[0])]),
        0.5*(ell_arr[1:] + ell_arr[:-1]),
        jnp.array([ell_arr[-1] + 0.5*(ell_arr[-1]-ell_arr[-2])])
    ))
    delta_ell = edges[1:] - edges[:-1]  # shape: (n_bins,)
    
    # 6) Construct the diagonal (Gaussian) term.
    #    Note: The factor of 2/(2l+1) is divided by Delta_ell (i.e. bin width) and f_sky.
    diag_term = 2.0 * (C_yy + noise_ell)**2 / ((2.0 * ell_arr + 1.0) * delta_ell * f_sky)
    # diag_term = 2.0 * (C_yy + noise_ell)**2 / ((2.0 * ell_arr + 1.0) * f_sky)
    M_G = jnp.diag(diag_term)
    
    # 7) Add the trispectrum contribution (scaled by 1/(4pi f_sky))
    M = M_G + T_ell_ellprime / (4.0 * jnp.pi * f_sky)

    
    return ell_arr, M, M_G



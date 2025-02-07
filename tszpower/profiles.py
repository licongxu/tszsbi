import numpy as np
import jax
import jax.numpy as jnp
import functools
import jax.scipy as jscipy
from jax.scipy.interpolate import RegularGridInterpolator
import mcfit
from mcfit import TophatVar
from .utils import get_ell_range


# Import the shared classy_sz from the package
from . import classy_sz

def gnfw_pressure_profile(x, z, m, params_values_dict = None):
    """
    GNFW pressure profile
    """
    # Obtain cosmological parameters from classy_sz
    rparams = classy_sz.get_all_relevant_params(params_values_dict = params_values_dict)
    # rparams = classy_sz.pars
    conv_fac = 299792.458 # speed of light
    h = rparams['H0']/100
    H = classy_sz.get_hubble_at_z(z, params_values_dict = params_values_dict) * conv_fac  # Hubble parameter at given z
    H0 = rparams['H0']  # Hubble parameter at z=0

    # GNFW prefactor
    B = rparams['B']    
    m_delta_tilde = (m / B) # convert to M_sun 
    C = 1.65 * (h / 0.7)**2 * (H / H0)**(8 / 3) * (m_delta_tilde / (0.7 * 3e14))**(2 / 3 + 0.12) * (0.7/h)**1.5 # eV cm^-3
    c500 = rparams['c500']
    gamma = rparams['gammaGNFW']
    alpha = rparams['alphaGNFW']
    beta = rparams['betaGNFW']
    P0 = rparams['P0GNFW']

    # Calculate scaled radius and pressure profile
    scaled_x = c500 * x
    term1 = scaled_x**(-gamma)
    term2 = (1 + scaled_x**alpha)**((gamma - beta) / alpha)
    Pe = C * P0 * term1 * term2

    return Pe


def window_function(x, x_min, x_max):
    """
    Since the integral is between x_min and x_max
    """

    return jnp.where((x >= x_min) & (x <= x_max), 1.0, 0.0)

def hankel_integrand(x, z, m, x_min=1e-6, x_max=4, params_values_dict = None):
    """
    Compute x^0.5 * Pe(x) * W(x).
    Handles x and m as arrays using vmap for vectorization.
    """
    # Vectorize over m
    def single_m(m_val, params_values_dict = params_values_dict):
        # Compute GNFW pressure profile for given m
        Pe = gnfw_pressure_profile(x, z, m_val, params_values_dict = params_values_dict)
        
        # Compute the window function
        W_x = window_function(x, x_min, x_max)
        
        # Combine the result
        return x**0.5 * Pe * W_x
    
    # Apply vmap to vectorize over m
    result = jax.vmap(single_m)(m)
    
    return result  # Shape: (len(m), len(x))

def mpc_per_h_to_cm(mpc_per_h, h):
    """
    Converts a distance in Mpc/h to centimeters.
    """
    # 1 megaparsec (Mpc) in centimeters
    Mpc_to_cm = 3.085677581e24  # cm

    # Convert Mpc/h to Mpc by dividing by h
    mpc = mpc_per_h / h

    # Convert Mpc to cm
    cm = mpc * Mpc_to_cm

    return cm

def y_ell_prefactor(z, m, delta = 500, params_values_dict = None): 
    
    rparams = classy_sz.get_all_relevant_params(params_values_dict = params_values_dict)
    # rparams = classy_sz.pars
    h = rparams['H0']/100
    B = rparams['B']
    
    # print(rparams)
    me_in_eV = 510998.95 # electron mass in eV/c^2
    # me_in_eV = 511000 # electron mass in eV/c^2
    sigmat_cm = 6.6524587321e-25 # Thomson cross section in cm^2
    sigmat_over_mec2 = sigmat_cm / me_in_eV
    # sigmat_over_mec2 = 283.2980000259841 / 0.5176

    dAz = classy_sz.get_angular_distance_at_z(z,params_values_dict = params_values_dict)*h
    # dAz = classy_sz.get_angular_distance_at_z(z,params_values_dict = params_values_dict)/(1+z)*h # in Mpc/h
    r_delta = classy_sz.get_r_delta_of_m_delta_at_z(delta, m, z, params_values_dict = params_values_dict)/(B**(1/3))  # in Mpc/h
    ell_delta = dAz/r_delta
    # print(ell_delta)
    h = rparams['H0']/100

    r_delta_cm = mpc_per_h_to_cm(r_delta, h)  # r is in cm 
    # print(r_delta_cm/ell_delta**2)
    # print(sigmat_over_mec2)

    prefactor = sigmat_over_mec2 * 4 * np.pi * r_delta_cm/(ell_delta**2)

    return prefactor

# # Define x in logarithmic space
# x_array = np.logspace(jnp.log10(1e-6), jnp.log10(6e2), num=256)  # Avoid x = 0 to prevent divergence
# # Hankel transform with JAX
# H = mcfit.Hankel(x_array, nu=0.5, lowring=True, backend='jax') 
# H_jit = jax.jit(functools.partial(H, extrap=False))

def y_ell_complete(z, m, x_min=1e-6, x_max=4, params_values_dict = None):

    rparams = classy_sz.get_all_relevant_params(params_values_dict = params_values_dict)
    h = rparams['H0']/100
    B = rparams['B']
    
    prefactor = y_ell_prefactor(z, m, params_values_dict=params_values_dict)
    # print(prefactor)
    # Define x in logarithmic space
    x = jnp.logspace(jnp.log10(1e-6), jnp.log10(6e2), num=128)  # Avoid x = 0 to prevent divergence
    x = jnp.array(x)

    integrand = hankel_integrand(x, z, m, x_min=1e-6, x_max=4, params_values_dict=params_values_dict)
    # print(integrand)

    # Hankel transform with JAX
    H = mcfit.Hankel(x, nu=0.5, lowring=True, backend='jax') 
    H_jit = jax.jit(functools.partial(H, extrap=False))

    k, y_k = H_jit(integrand) # Note that k = ell/ell_delta
    # print(k)
    # print(y_k)

    dAz = classy_sz.get_angular_distance_at_z(z, params_values_dict = params_values_dict) * h # in Mpc/h
    # dAz = classy_sz.get_angular_distance_at_z(z,params_values_dict = params_values_dict)/(1+z)*h # in Mpc/h

    delta = 500
    r_delta = classy_sz.get_r_delta_of_m_delta_at_z(delta, m, z, params_values_dict = params_values_dict)/(B**(1/3))  # in Mpc/h
    # print(r_delta)
    ell_delta = dAz/r_delta
    # print(ell_delta)

    ell = jnp.zeros(ell_delta.shape)
    # ell = k * ell_delta  # Note that k = ell/ell_delta
    ell = k[None, :] * ell_delta[:, None]
    # print("shape  of ell:", ell.shape)


    # print("shape of k:", k.shape)
    # print("shape of ell_delta:", ell_delta.shape)

    # print(ell)
    # print(ell_delta[:, None].shape)
    # print(k[None, :])
    # print(y_k.shape)
    # print(prefactor.shape)
    # print(prefactor[:, None])

    # y_ell = prefactor * y_k * np.sqrt(np.pi/(2*k)) # multiply the prefactor of spherical Bessel to Hankel
    y_ell = prefactor[:, None] * y_k * jnp.sqrt(jnp.pi / (2 * k[None, :]))

    return ell, y_ell

def y_ell_interpolate(z, m, params_values_dict = None):
    """
    Interpolate y_ell values onto a uniform ell grid for multiple m values.
    """
    # Get cosmological parameters
    # rparams = classy_sz.pars
    # l_min = rparams['ell_min']
    # l_max = rparams['ell_max']
    # dlogell = rparams['dlogell']


    # log10_l_min = jnp.log10(l_min)
    # log10_l_max = jnp.log10(l_max)
    # num = jnp.array(((log10_l_max - log10_l_min) / dlogell) + 1, int)
    # print(num)

    # Compute the complete y_ell values
    ell_nointer_list, y_ell_nointer_list = y_ell_complete(z, m, params_values_dict = params_values_dict)

    # Define evaluation ell values (uniform grid)
    # ell_eval = jnp.logspace(log10_l_min, log10_l_max, num=num)
    ell_eval = get_ell_range()

    # Interpolator function for a single m
    def interpolate_single(ell_nointer, y_ell_nointer):
        # interpolator = RegularGridInterpolator((ell_nointer,), y_ell_nointer, method='linear', bounds_error=False, fill_value=None)
        interpolator = RegularGridInterpolator((ell_nointer,), y_ell_nointer, method='linear', bounds_error=False, fill_value=None)
        return interpolator(ell_eval)

    # Vectorize the interpolation across all m
    interpolate_all = jax.vmap(interpolate_single, in_axes=(0, 0), out_axes=0)

    # Perform the interpolation
    y_ell_eval_list = interpolate_all(ell_nointer_list, y_ell_nointer_list)
    # print(ell_eval)

    return ell_eval, y_ell_eval_list


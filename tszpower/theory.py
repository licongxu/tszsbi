# tszpower/theory.py

import numpy as np
import jax.numpy as jnp
import tszpower  # assuming tszpower already provides these functions
import time

# You can either mimic the structure of your Cobaya theory classes
# or simply define functions. Here is one possible approach:

def compute_sz_power(allparams):
    """
    Computes the tSZ power spectrum (1-halo term) for the given cosmological parameters.
    
    Parameters
    ----------
    cosmology_params : dict
        Dictionary with keys "omega_b", "omega_cdm", "H0", "tau_reio",
        "ln10_10A_s", "n_s", and "B".
    
    Returns
    -------
    ell : ndarray
        Multipole moments.
    cl_1h : ndarray
        tSZ power spectrum (1-halo term).
    """
    # Fixed astrophysical parameters
    fixed_params = {
        "M_min": 1e10,
        "M_max": 3.5e15,
        "z_min": 0.005,
        "z_max": 3.0,
        "P0GNFW": 8.130,
        "c500": 1.156,
        "gammaGNFW": 0.3292,
        "alphaGNFW": 1.0620,
        "betaGNFW": 5.4807,
        "jax": 1,
        "cosmo_model": 1,
        # "omega_b": 0.0224,
        # "omega_cdm": 0.119,
        # "H0": 69,
        "tau_reio": 0.06,
        # "ln10_10A_s": 3.044,
        # "n_s": 0.9645,

    }
    # Build the parameter dictionary expected by tszpower (note the key for A_s)
    updated_params = {
        "omega_b": allparams["omega_b"],
        "omega_cdm": allparams["omega_cdm"],
        "H0": allparams["H0"],
        # "tau_reio": allparams["tau_reio"],
        "ln10^{10}A_s": allparams["ln10_10A_s"],
        "n_s": allparams["n_s"],
        "B": allparams["B"],
    }
    # print(updated_params)
    updated_params.update(fixed_params)
    # print(updated_params)
    
    # Retrieve ell array and compute the power spectrum
    ell = tszpower.get_ell_range()  # This function should already be in your package.
    # print(updated_params)
    cl_total = tszpower.compute_integral(params_values_dict=updated_params)
    # Convert units as in your original code

    cl_total = cl_total * ell * (ell + 1) / (2 * jnp.pi) * 1e12
    
    # For this example we assume only a 1-halo contribution:
    cl_1h = cl_total
    return ell, cl_1h


def compute_foreground(allparams, fg_template_path='data/data_fg-ell-cib_rs_ir_cn-total-planck-collab-15.txt'):
    """
    Computes the foreground contribution given nuisance parameters.
    
    Parameters
    ----------
    nuisance_params : dict
        Dictionary with keys "A_cib", "A_ir", "A_rs".
    fg_template_path : str
        Path to the foreground template file.
    
    Returns
    -------
    cl_fg : ndarray
        Foreground power spectrum.
    """
    # Load the foreground template.
    # Expected columns: ell, CIB template, RS template, IR template, CN template.
    D_fg = np.loadtxt(fg_template_path)
    fg_ell = D_fg[:, 0]
    A_CIB_MODEL = D_fg[:, 1]
    A_RS_MODEL  = D_fg[:, 2]
    A_IR_MODEL  = D_fg[:, 3]
    A_CN_MODEL  = D_fg[:, 4]
    
    # Extract nuisance parameters
    A_cib = allparams["A_cib"]
    A_rs = allparams["A_rs"]
    A_ir = allparams["A_ir"]
    # Here A_CN is fixed by calibration
    A_cn = 0.9033
    
    cl_fg = A_cib * A_CIB_MODEL + A_rs * A_RS_MODEL + A_ir * A_IR_MODEL + A_cn * A_CN_MODEL
    return cl_fg

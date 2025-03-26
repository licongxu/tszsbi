from .tsz import compute_integral, compute_trispectrum, compute_tsz_covariance
from .utils import get_ell_range
import jax.numpy as jnp

def compute_Cell_yy(params_value_dict = None):
    
    ell = get_ell_range()
    C_ell_yy = compute_integral(params_values_dict=params_value_dict)

    return C_ell_yy


def compute_Dell_yy(params_value_dict = None):

    ell = get_ell_range()
    C_ell_yy = compute_integral(params_values_dict=params_value_dict)
    D_ell_yy = ell*(ell+1)*C_ell_yy/(2*jnp.pi)*1e12

    return D_ell_yy

def compute_Dell_yy_with_error(params_value_dict = None):

    ell, M, M_G = compute_tsz_covariance(params_values_dict=params_value_dict, noise_ell=None, f_sky=1.0)

    C_ell_yy = compute_integral(params_values_dict=params_value_dict)
    D_ell_yy = ell*(ell+1)*C_ell_yy/(2*jnp.pi)*1e12

    diag_M_full = jnp.diag(M)
    sigma_full = jnp.sqrt(diag_M_full)
    sigma_full = ell*(ell+1)*sigma_full/(2*jnp.pi)*1e12

    diag_M_gauss = jnp.diag(M_G)
    sigma_gauss = jnp.sqrt(diag_M_gauss)
    sigma_gauss = ell*(ell+1)*sigma_gauss/(2*jnp.pi)*1e12

    return D_ell_yy, sigma_full, sigma_gauss
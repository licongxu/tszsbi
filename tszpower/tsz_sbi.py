import tszpower
import jax
import jax.numpy as jnp
import jax.random as random
from jax import grad
import matplotlib.pyplot as plt
import numpy as np
from .utils import get_ell_range
from .tsz import compute_integral, compute_tsz_covariance
from . import classy_sz  # shared instance

# --- Helper Functions for Broadcasting ---

def ensure_array(arg):
    """Ensure that the argument is a JAX array."""
    if not isinstance(arg, jnp.ndarray):
        return jnp.array(arg)
    return arg

def broadcast_to_batch(arg, batch_size):
    """If arg is scalar (rank 0), broadcast it to shape (batch_size,)."""
    arg = ensure_array(arg)
    if arg.ndim == 0:
        return jnp.broadcast_to(arg, (batch_size,))
    return arg

def get_batch_size(*args):
    """
    Determine the batch size from the first argument that is batched.
    If none are batched, return None.
    """
    for arg in args:
        arr = ensure_array(arg)
        if arr.ndim > 0:
            return arr.shape[0]
    return None

# --- Function 1: compute_Cl_yy_noiseless ---

def compute_Cl_yy_noiseless(logA, omega_b, omega_cdm, H0, n_s, B, params_values_dict=None):
    """
    Differentiable forward model for C_ell^yy that accepts scalar or batched inputs
    for six cosmological parameters.
    """
    ell = get_ell_range()  # jnp.array of shape (n_ell,)

    # Scalar version: all six parameters are passed explicitly.
    def compute_for_params(logA_i, omega_b_i, omega_cdm_i, H0_i, n_s_i, B_i):
        pars = classy_sz.get_all_relevant_params(params_values_dict=params_values_dict)
        pars.update({
            'ln10^{10}A_s': logA_i,
            'omega_b':      omega_b_i,
            'omega_cdm':    omega_cdm_i,
            'H0':           H0_i,
            'n_s':          n_s_i,
            'B':            B_i
        })
        Cl_theory = compute_integral(params_values_dict=pars)
        return Cl_theory

    batch_size = get_batch_size(logA, omega_b, omega_cdm, H0, n_s, B)
    if batch_size is None:
        return compute_for_params(logA, omega_b, omega_cdm, H0, n_s, B)
    else:
        # Broadcast any scalar inputs to match the batch size.
        logA      = broadcast_to_batch(logA, batch_size)
        omega_b   = broadcast_to_batch(omega_b, batch_size)
        omega_cdm = broadcast_to_batch(omega_cdm, batch_size)
        H0        = broadcast_to_batch(H0, batch_size)
        n_s       = broadcast_to_batch(n_s, batch_size)
        B         = broadcast_to_batch(B, batch_size)
        # Vectorize the scalar function over the batch dimension.
        vectorized_compute = jax.vmap(compute_for_params, in_axes=(0, 0, 0, 0, 0, 0))
        return vectorized_compute(logA, omega_b, omega_cdm, H0, n_s, B)

# --- Function 2: compute_Nl_yy ---

def compute_Nl_yy(logA, omega_b, omega_cdm, H0, n_s, B, key, params_values_dict=None):
    """
    Generates a noise realization for C_ell^yy given cosmological parameters.
    """
    ell = get_ell_range()  # jnp.array of shape (n_ell,)

    # Scalar version.
    def compute_for_params(logA_i, omega_b_i, omega_cdm_i, H0_i, n_s_i, B_i):
        pars = classy_sz.get_all_relevant_params(params_values_dict=params_values_dict)
        pars.update({
            'ln10^{10}A_s': logA_i,
            'omega_b':      omega_b_i,
            'omega_cdm':    omega_cdm_i,
            'H0':           H0_i,
            'n_s':          n_s_i,
            'B':            B_i
        })
        Mllp_theory = compute_tsz_covariance(params_values_dict=pars)[1]
        L = jnp.linalg.cholesky(Mllp_theory)
        # Use the provided key (assumed scalar) to generate noise.
        z = jax.random.normal(key, shape=(Mllp_theory.shape[0],))
        N_l_sim = L @ z
        return N_l_sim

    batch_size = get_batch_size(logA, omega_b, omega_cdm, H0, n_s, B)
    if batch_size is None:
        return compute_for_params(logA, omega_b, omega_cdm, H0, n_s, B)
    else:
        logA      = broadcast_to_batch(logA, batch_size)
        omega_b   = broadcast_to_batch(omega_b, batch_size)
        omega_cdm = broadcast_to_batch(omega_cdm, batch_size)
        H0        = broadcast_to_batch(H0, batch_size)
        n_s       = broadcast_to_batch(n_s, batch_size)
        B         = broadcast_to_batch(B, batch_size)
        vectorized_compute = jax.vmap(compute_for_params, in_axes=(0, 0, 0, 0, 0, 0))
        return vectorized_compute(logA, omega_b, omega_cdm, H0, n_s, B)

# --- Function 3: compute_Cl_yy ---

def compute_Cl_yy(logA, omega_b, omega_cdm, H0, n_s, B, key, params_values_dict=None, n_realizations=1):
    """
    Computes the full C_ell^yy (theory plus noise) for given parameters.
    If n_realizations > 1, returns an array with an extra leading dimension.
    """
    Cl_noiseless = compute_Cl_yy_noiseless(logA, omega_b, omega_cdm, H0, n_s, B,
                                            params_values_dict=params_values_dict)
    if n_realizations == 1:
        Nl_yy = compute_Nl_yy(logA, omega_b, omega_cdm, H0, n_s, B, key,
                              params_values_dict=params_values_dict)
        return Cl_noiseless + Nl_yy
    else:
        keys = jax.random.split(key, n_realizations)
        # For multiple realizations, we assume cosmological parameters are scalar or uniformly batched.
        Nl_yy = jax.vmap(lambda k: compute_Nl_yy(logA, omega_b, omega_cdm, H0, n_s, B, k,
                                                  params_values_dict=params_values_dict))(keys)
        # Cl_noiseless has shape (n_ell,); add a new axis for broadcasting.
        return Cl_noiseless[None, :] + Nl_yy

# --- Function 4: compute_foreground ---

def compute_foreground(A_cib, A_rs, A_ir, fg_template_path='data/data_fg-ell-cib_rs_ir_cn-total-planck-collab-15.txt'):
    """
    Computes the foreground contribution given nuisance parameters.
    If the nuisance parameters are batched, the output is batched.
    """
    # Load the template.
    D_fg = np.loadtxt(fg_template_path)
    fg_ell = D_fg[:, 0]
    A_CIB_MODEL = jnp.array(D_fg[:, 1])
    A_RS_MODEL  = jnp.array(D_fg[:, 2])
    A_IR_MODEL  = jnp.array(D_fg[:, 3])
    A_CN_MODEL  = jnp.array(D_fg[:, 4])
    A_cn = 0.9033

    batch_size = get_batch_size(A_cib, A_rs, A_ir)
    if batch_size is None:
        # Assume nuisance parameters are scalars.
        return A_cib * A_CIB_MODEL + A_rs * A_RS_MODEL + A_ir * A_IR_MODEL + A_cn * A_CN_MODEL
    else:
        # Broadcast nuisance parameters to (batch, 1) so they multiply properly with (1, n_ell) templates.
        A_cib = broadcast_to_batch(A_cib, batch_size)[:, None]
        A_rs  = broadcast_to_batch(A_rs, batch_size)[:, None]
        A_ir  = broadcast_to_batch(A_ir, batch_size)[:, None]
        return (A_cib * A_CIB_MODEL[None, :] +
                A_rs  * A_RS_MODEL[None, :] +
                A_ir  * A_IR_MODEL[None, :] +
                A_cn  * A_CN_MODEL[None, :])

# --- Function 5: compute_Cl_yy_total ---

def compute_Cl_yy_total(logA, omega_b, omega_cdm, H0, n_s, B,
                        A_cib, A_rs, A_ir, key, params_values_dict=None, n_realizations=1):
    """
    Computes the total C_ell^yy as the sum of the theoretical (noiseless) component,
    the foreground contribution, and one or more noise realizations.
    """
    Cl_noiseless = compute_Cl_yy_noiseless(logA, omega_b, omega_cdm, H0, n_s, B,
                                            params_values_dict=params_values_dict)
    cl_fg = compute_foreground(A_cib, A_rs, A_ir)
    ell = get_ell_range()
    if n_realizations == 1:
        Nl_yy = compute_Nl_yy(logA, omega_b, omega_cdm, H0, n_s, B, key,
                              params_values_dict=params_values_dict)
        return (Cl_noiseless + Nl_yy) * (ell * (ell + 1) / (2 * jnp.pi)) + cl_fg
    else:
        keys = jax.random.split(key, n_realizations)
        Nl_yy = jax.vmap(lambda k: compute_Nl_yy(logA, omega_b, omega_cdm, H0, n_s, B, k,
                                                  params_values_dict=params_values_dict))(keys)
        return (Cl_noiseless[None, :] + Nl_yy) * (ell * (ell + 1) / (2 * jnp.pi)) + cl_fg[None, :]

# --- Example usage ---

# Convert lists to jnp.array to ensure batched inputs.
# logA      = jnp.array([3., 3.2])
# omega_b   = jnp.array([0.0225, 0.0235])
# omega_cdm = jnp.array([0.12, 0.13])
# H0        = jnp.array([69., 71.])
# n_s       = jnp.array([0.965, 0.967])
# B         = jnp.array([1.6, 2.0])
# A_cib     = jnp.array([0.5, 1.5])
# A_rs      = jnp.array([1.5, 2.5])
# A_ir      = jnp.array([1.5, 2.5])
# key       = jax.random.PRNGKey(66)

# # Compute components.
# Cl_noiseless = compute_Cl_yy_noiseless(logA, omega_b, omega_cdm, H0, n_s, B, params_values_dict=allpars)
# cl_fg = compute_foreground(A_cib, A_rs, A_ir)
# Cl_total = compute_Cl_yy_total(logA, omega_b, omega_cdm, H0, n_s, B, A_cib, A_rs, A_ir,
#                                key, params_values_dict=allpars, n_realizations=1)

# print("Cl_noiseless shape:", Cl_noiseless.shape)
# print("Foreground shape:", cl_fg.shape)
# print("Cl_total shape:", Cl_total.shape)

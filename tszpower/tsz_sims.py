import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import torch
from .utils import get_ell_range, get_batch_size, broadcast_to_batch
from .power_spectra import compute_Dell_yy
from . import classy_sz  # shared instance

# --- Function 1: compute_Cl_yy_noiseless ---

def compute_Dl_yy_noiseless(logA, omega_b, omega_cdm, H0, n_s, B, params_values_dict=None):
    """
    Differentiable forward model for C_ell^yy that accepts scalar or batched inputs
    for six cosmological parameters.
    """
    ell = get_ell_range()  # jnp.array of shape (n_ell,)

    # Scalar version: all six parameters are passed explicitly.
    def compute_for_params(logA_i, omega_b_i, omega_cdm_i, H0_i, n_s_i, B_i):
        pars = classy_sz.get_all_relevant_params(params_values_dict=params_values_dict)

        # Ensure only numeric parameters are used
        pars = {k: v for k, v in pars.items() if isinstance(v, (int, float, np.ndarray))}
        pars.update({
            'ln10^{10}A_s': logA_i,
            'omega_b':      omega_b_i,
            'omega_cdm':    omega_cdm_i,
            'H0':           H0_i,
            'n_s':          n_s_i,
            'B':            B_i
        })
        # print(pars)
        Dl_theory = compute_Dell_yy(params_value_dict=pars)
        return Dl_theory


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
    def compute_for_params(logA_i, omega_b_i, omega_cdm_i, H0_i, n_s_i, B_i, data_path = "data/data_ps-ell-y2-erry2_total-planck-collab-15.txt"):
        pars = classy_sz.get_all_relevant_params(params_values_dict=params_values_dict)
        pars.update({
            'ln10^{10}A_s': logA_i,
            'omega_b':      omega_b_i,
            'omega_cdm':    omega_cdm_i,
            'H0':           H0_i,
            'n_s':          n_s_i,
            'B':            B_i
        })
        # Mllp_theory = compute_tsz_covariance(params_values_dict=pars)[1]

        D = np.loadtxt(data_path)
        # Mllp_theory = jnp.diag(((D[:, 2])/(ell*(ell+1)*1e12/(2*jnp.pi)))**2) # This is just gaussian covariance
        Mllp_scaled = jnp.diag((D[:, 2])**2)

        L = jnp.linalg.cholesky(Mllp_scaled)
        # Use the provided key (assumed scalar) to generate noise.
        z = jax.random.normal(key, shape=(Mllp_scaled.shape[0],))
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

def compute_Dl_yy(logA, omega_b, omega_cdm, H0, n_s, B, key, params_values_dict=None, n_realizations=1):
    """
    Computes the full D_ell^yy (theory plus noise) for given parameters.
    If n_realizations > 1, returns an array with an extra leading dimension.
    """
    Dl_noiseless = compute_Dl_yy_noiseless(logA, omega_b, omega_cdm, H0, n_s, B,
                                            params_values_dict=params_values_dict)
    if n_realizations == 1:
        Nl_yy = compute_Nl_yy(logA, omega_b, omega_cdm, H0, n_s, B, key,
                              params_values_dict=params_values_dict)
        return Dl_noiseless + Nl_yy
    else:
        keys = jax.random.split(key, n_realizations)
        # For multiple realizations, we assume cosmological parameters are scalar or uniformly batched.
        Nl_yy = jax.vmap(lambda k: compute_Nl_yy(logA, omega_b, omega_cdm, H0, n_s, B, k,
                                                  params_values_dict=params_values_dict))(keys)
        ell = get_ell_range()
        # Cl_noiseless has shape (n_ell,); add a new axis for broadcasting.
        return Dl_noiseless[None, :] + (Nl_yy)

# --- Function 4: compute_foreground ---

def compute_dl_foreground(A_cib, A_rs, A_ir, fg_template_path='data/data_fg-ell-cib_rs_ir_cn-total-planck-collab-15.txt'):
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
@jax.jit
def compute_Dl_yy_total(logA, omega_b, omega_cdm, H0, n_s, B,
                        A_cib, A_rs, A_ir, key, params_values_dict=None, n_realizations=1):
    """
    Computes the total C_ell^yy as the sum of the theoretical (noiseless) component,
    the foreground contribution, and one or more noise realizations.
    """
    Dl_noiseless = compute_Dl_yy_noiseless(logA, omega_b, omega_cdm, H0, n_s, B,
                                            params_values_dict=params_values_dict)
    dl_fg = compute_dl_foreground(A_cib, A_rs, A_ir)
    # ell = get_ell_range()

    #TODO: Fix this to include the noise realization using jax.lax.con()
    # if n_realizations == 1:
    #     Nl_yy = compute_Nl_yy(logA, omega_b, omega_cdm, H0, n_s, B, key,
    #                           params_values_dict=params_values_dict)
    #     return (Cl_noiseless + Nl_yy) * (ell * (ell + 1)*1e12 / (2 * jnp.pi)) + cl_fg
    # else:
    #     keys = jax.random.split(key, n_realizations)
    #     Nl_yy = jax.vmap(lambda k: compute_Nl_yy(logA, omega_b, omega_cdm, H0, n_s, B, k,
    #                                               params_values_dict=params_values_dict))(keys)
    #     return (Cl_noiseless[None, :] + Nl_yy) * (ell * (ell + 1)*1e12 / (2 * jnp.pi))  + cl_fg[None, :]
    Nl_yy = compute_Nl_yy(logA, omega_b, omega_cdm, H0, n_s, B, key,
                              params_values_dict=params_values_dict)
    return Dl_noiseless + Nl_yy + dl_fg


def simulator(theta: torch.Tensor, params_value_dict=None) -> torch.Tensor:
    # Ensure theta is 2D (batch, 9)
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)
    batch_size = theta.shape[0]
    
    # Generate a base key and split it for each simulation.
    base_key = jax.random.PRNGKey(12083)
    keys = jax.random.split(base_key, batch_size)
    
    sim_list = []
    for i in range(batch_size):
        # Extract each parameter as a Python float
        logA      = float(theta[i, 0])
        omega_b   = float(theta[i, 1])
        omega_cdm = float(theta[i, 2])
        H0        = float(theta[i, 3])
        n_s       = float(theta[i, 4])
        B         = float(theta[i, 5])
        A_cib     = float(theta[i, 6])
        A_rs      = float(theta[i, 7])
        A_ir      = float(theta[i, 8])
        
        # Call your tszpower simulator (which uses JAX internally)
        sim_i = compute_Dl_yy_total(
            logA,
            omega_b,
            omega_cdm,
            H0,
            n_s,
            B,
            A_cib,
            A_rs,
            A_ir,
            keys[i],
            params_values_dict=params_value_dict,  # your global parameter dictionary
            n_realizations=1
        )
        # Convert the returned JAX array to a NumPy array and then to a torch.Tensor
        sim_torch = torch.tensor(np.array(sim_i), dtype=torch.float32)
        sim_list.append(sim_torch)

    # Stack the results to form a tensor of shape (batch, n_ell)
    return torch.stack(sim_list, dim=0)

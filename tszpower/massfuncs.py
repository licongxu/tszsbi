import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from mcfit import TophatVar
from . import classy_sz

# -----------------------------------------------------------------------------
# Precompute a default k-grid and pre-create a single TophatVar instance.
# This instance is constructed with a fixed k-grid (from z=1 and default params)
# and then used for all subsequent calls (including for numerical derivatives).
# -----------------------------------------------------------------------------
_default_params = classy_sz.get_all_relevant_params()
_, _ks = classy_sz.get_pkl_at_z(1., params_values_dict=_default_params)
# Pre-create the TophatVar instance (note: we are not using deriv=1 now)
_tophat_instance = TophatVar(_ks, lowring=True, backend='jax')


def MF_T08(sigmas, z, delta_mean):
    # Convert delta_mean to log scale
    delta_mean = jnp.log10(delta_mean)
    
    # Define parameters as JAX arrays
    delta_mean_tab = jnp.log10(jnp.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200]))
    A_tab = jnp.array([0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260])
    aa_tab = jnp.array([1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66])
    b_tab = jnp.array([2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41])
    c_tab = jnp.array([1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44])

    # Linear interpolation using jnp.interp
    Ap = jnp.interp(delta_mean, delta_mean_tab, A_tab) * (1 + z) ** -0.14
    a = jnp.interp(delta_mean, delta_mean_tab, aa_tab) * (1 + z) ** -0.06
    b = jnp.interp(delta_mean, delta_mean_tab, b_tab) * (1 + z) ** -jnp.power(10, -jnp.power(0.75 / jnp.log10(jnp.power(10, delta_mean) / 75), 1.2))
    c = jnp.interp(delta_mean, delta_mean_tab, c_tab)
    
    # Calculate final result
    result = 0.5 * Ap[:, None] * (jnp.power(sigmas / b[:, None], -a[:, None]) + 1) * jnp.exp(-c[:, None] / sigmas**2)
    return result


def get_hmf_grid(delta=500, delta_def='critical', params_values_dict=None):
    # Use provided params or default ones.
    # NOTE: The k-grid is fixed from the module-level _ks.
    rparams = classy_sz.get_all_relevant_params(params_values_dict=params_values_dict or _default_params)
    h = rparams['h']
    
    # Use the fixed k-grid from the module level.
    ks = _ks

    # Define a function for obtaining the power spectrum at a given redshift.
    def get_pks_for_z(zp):
        pks, _ = classy_sz.get_pkl_at_z(zp, params_values_dict=params_values_dict or _default_params)
        return pks.flatten()

    # Get the redshift grid and evaluate power spectra.
    z_grid = classy_sz.z_grid()
    P = jax.vmap(get_pks_for_z)(z_grid).T  # shape: (n_k, n_z)

    # Compute the variance using the precomputed _tophat_instance.
    var = jax.vmap(lambda pks: _tophat_instance(pks, extrap=True)[1], in_axes=1)(P)

    # Compute the numerical derivative (of the variance) using the precomputed instance.
    def compute_numerical_dvar(pks):
        rvar, var_z = _tophat_instance(pks, extrap=True)
        # Compute derivative numerically: d/dR sqrt(var_z) then scale back.
        dvar_z = jnp.gradient(jnp.sqrt(var_z), rvar) * 2. * jnp.sqrt(var_z)
        return dvar_z

    dvar = jax.vmap(compute_numerical_dvar, in_axes=1)(P)

    # Compute the smoothing scale from the first column of P.
    R, _ = _tophat_instance(P[:, 0], extrap=True)
    R = R.flatten()  # Ensure R is 1D.
    lnx_grid = jnp.log(1 + z_grid)
    # (lnr_grid is available if needed: lnr_grid = jnp.log(R))

    lnsigma_grid = 0.5 * jnp.log(var)
    dsigma2_grid = dvar

    Rh = R * rparams['h']
    lnm_grid = jnp.log(4 * jnp.pi * rparams['Omega0_cb'] * rparams['Rho_crit_0'] * Rh**3 / 3.)  # in h-units

    # Choose delta_mean based on the definition.
    if delta_def == 'critical':
        delta_mean = classy_sz.get_delta_mean_from_delta_crit_at_z(delta, z_grid, params_values_dict=params_values_dict or _default_params)
    elif delta_def == 'mean':
        delta_mean = jnp.full_like(z_grid, delta)
    else:
        raise ValueError("delta_def must be either 'critical' or 'mean'.")

    delta_c = (3. / 20.) * jnp.power(12. * jnp.pi, 2. / 3.)
    sigmas = jnp.exp(lnsigma_grid)
    hmf = MF_T08(sigmas, z_grid, delta_mean)

    lnSigma2 = 2. * lnsigma_grid
    dlnsigmadlnR = dsigma2_grid / 2.
    dlnSigma2dlnR = 2. * dlnsigmadlnR * R / jnp.exp(lnSigma2)
    dlnnudlnRh = -dlnSigma2dlnR

    # Return dn/dlogM in units of h^3 Mpc^-3.
    dndlnm_grid = (1. / 3.) * (3. / (4. * jnp.pi * Rh**3)) * dlnnudlnRh * hmf
    return lnx_grid, lnm_grid, dndlnm_grid


# def get_hmf_at_z_and_m(z,m,params_values_dict = None):
#     lnx, lnm, dndlnm = get_hmf_grid(delta = 500, delta_def = 'critical', params_values_dict = params_values_dict)
#     hmf_interp = jscipy.interpolate.RegularGridInterpolator((lnx, lnm), jnp.log(dndlnm))
#     lnxp = jnp.log(1.+z)
#     lnmp = jnp.log(m)
#     return jnp.exp(hmf_interp((lnxp,lnmp)))
def get_hmf_at_z_and_m(z,m,params_values_dict = None, eps = 1e-12):
    lnx, lnm, dndlnm = get_hmf_grid(delta = 500, delta_def = 'critical', params_values_dict = params_values_dict)
    # hmf_interp = jscipy.interpolate.RegularGridInterpolator((lnx, lnm), jnp.log(dndlnm))
    hmf_interp = jscipy.interpolate.RegularGridInterpolator((lnx, lnm), jnp.log(jnp.clip(dndlnm, a_min=eps)))
    lnxp = jnp.log(1.+z)
    lnmp = jnp.log(m)
    return jnp.exp(hmf_interp((lnxp,lnmp)))
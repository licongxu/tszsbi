"""
Masked (unresolved) tSZ power spectrum with full completeness.

This module computes the theoretical prediction for the tSZ power spectrum
after masking resolved galaxy clusters above an SNR threshold, using the
*full* completeness function that accounts for two layers of scatter:

  1. **Intrinsic scatter** (log-normal in ln q with width sigma_lnY):
     Arises from the scatter in the Y-M relation (mass--observable relation).
     Modelled as a Gaussian in ln(q_m) centred on ln(q_bar), so that the
     true signal-to-noise q_m is drawn from a log-normal around the mean
     prediction q_bar(M, z).

  2. **Instrumental noise scatter** (unit-variance Gaussian on observed SNR):
     The observed q_obs = q_m + n, where n ~ N(0, 1).  A cluster enters the
     catalogue when q_obs >= q_cat.  For a given true q_m the detection
     probability is  P(q_obs >= q_cat | q_m) = [1 - erf((q_cat - q_m)/sqrt(2))] / 2.

These two effects are combined in a single convolution integral
(``completeness_convolution_jax``):

    P_det(M,z) = (1 / norm) * int dt  exp(-(t - mu)^2 / (2 sigma_lnY^2))
                                       * [1 - erf((q_cat - e^t) / sqrt(2))]

where mu = ln(q_bar(M,z)) and norm = 2 * sqrt(2 pi) * sigma_lnY.

The **unresolved** tSZ power spectrum keeps only the undetected fraction:

    C_ell^{unres} = int dz  int d(ln M)  [ y_ell^2  dn/d(ln M)  dV/dz/dOmega ]
                                          * [ 1 - P_det(M,z) ]

Public API
----------
- ``compute_masked_tsz_ps``          : C_ell  (continuous ell grid)
- ``compute_masked_tsz_Dell``        : D_ell = ell(ell+1) C_ell / (2 pi) * 1e12
- ``compute_masked_tsz_Dell_binned`` : D_ell binned to 18 Planck-style bands
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

# --- intra-package imports (building blocks already in tszpower) -----------
from . import classy_sz
from .utils import get_ell_range, simpson
from .tsz import get_integral_grid
from .maskedpower import (
    build_snr_grid,
    completeness_convolution_jax,
    bin_Dl_to_18,
)


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("n_grid", "nsig"))
def _pdet_grid(
    qbar_grid,
    *,
    sigma_lnY,
    q_cat,
    n_grid=4096,
    nsig=16.0,
):
    """
    Build the detection probability P_det(M, z) on the (z, m) grid,
    using the double-scatter completeness convolution.

    Parameters
    ----------
    qbar_grid : (n_z, n_m) jnp.ndarray
        Mean (theory) SNR on the integration grid.
    sigma_lnY : float
        Intrinsic log-normal scatter width.
    q_cat : float
        Catalogue SNR detection threshold.
    n_grid : int
        Number of quadrature points for the convolution integral.
    nsig : float
        Integration range in units of sigma_lnY around the mean.

    Returns
    -------
    P_det : (n_z, n_m) jnp.ndarray
        Detection probability at each (z, M) grid point.
    """
    qbar_flat = qbar_grid.reshape(-1)
    raw = jax.vmap(
        lambda qb: completeness_convolution_jax(
            qb, sigma_lnY, q_cat, n_grid=n_grid, nsig=nsig
        )
    )(qbar_flat)
    norm = 2.0 * jnp.sqrt(2.0 * jnp.pi) * sigma_lnY
    return (raw / norm).reshape(qbar_grid.shape)


@jax.jit
def _integrate_mz(integrand, z_grid, logm_grid):
    """
    Integrate the (n_z, n_m, n_ell) integrand over mass (dlnM) then
    redshift (dz) for each multipole slice, returning C_ell of shape (n_ell,).
    """
    n_ell = integrand.shape[-1]

    def scan_body(_, i):
        integrand_i = integrand[:, :, i]                     # (n_z, n_m)
        partial_m = simpson(integrand_i, x=logm_grid, axis=1)  # (n_z,)
        result    = simpson(partial_m,   x=z_grid,    axis=0)   # scalar
        return None, result

    _, C_yy = lax.scan(scan_body, None, jnp.arange(n_ell))
    return C_yy  # (n_ell,)


# ---------------------------------------------------------------------------
#  Public functions
# ---------------------------------------------------------------------------

@partial(
    jax.jit,
    static_argnames=(
        "sigma_obj_file",
        "skyfr_file",
        "filter_name",
        "theta_min",
        "theta_max",
        "n_grid",
        "nsig",
    ),
)
def compute_masked_tsz_ps(
    params_values_dict=None,
    *,
    q_cat,
    sigma_lnY,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
    n_grid=4096,
    nsig=16.0,
):
    """
    Compute the **unresolved** (masked) tSZ angular power spectrum C_ell
    using the full double-scatter completeness function.

    The completeness accounts for:
      - intrinsic log-normal scatter (sigma_lnY) in the Y--M relation
      - Gaussian instrumental noise scatter (unit variance in SNR space)

    The mask weight at each (M, z) is:

        W_unres(M, z) = 1 - P_det(M, z; q_cat, sigma_lnY)

    so that only the *undetected* (low-SNR) halos contribute to the
    residual power spectrum.

    Parameters
    ----------
    params_values_dict : dict
        Cosmological + scaling-relation parameters passed to classy_sz.
    q_cat : float
        Catalogue SNR detection threshold.
    sigma_lnY : float
        Intrinsic log-normal scatter width in the mass--observable relation.
    sigma_obj_file, skyfr_file : str
        Paths to the noise-curve and sky-fraction files.
    filter_name : str
        Matched-filter identifier (e.g. ``'immf6'``).
    theta_min, theta_max : float
        Angular-size range [arcmin] for the noise interpolation.
    n_grid : int
        Number of quadrature points for the completeness convolution.
    nsig : float
        Integration width (in units of sigma_lnY) for the convolution.

    Returns
    -------
    C_yy_unres : (n_ell,) jnp.ndarray
        Unresolved tSZ power spectrum C_ell^{yy} on the internal ell grid.
    """
    # --- 1. Reconstruct the (z, M) integration grids --------------------
    allparams = classy_sz.get_all_relevant_params(
        params_values_dict=params_values_dict
    )
    z_grid = jnp.geomspace(allparams["z_min"], allparams["z_max"], 100)
    m_grid = jnp.geomspace(allparams["M_min"], allparams["M_max"], 100)
    logm_grid = jnp.log(m_grid)

    # --- 2. Base (unmasked) integrand: y_ell^2 * dn/dlnM * dV/dz/dOmega
    integrand = get_integral_grid(
        params_values_dict=params_values_dict
    )  # (n_z, n_m, n_ell)

    # --- 3. Mean SNR field qbar(M, z) -----------------------------------
    qbar_grid = build_snr_grid(
        m_grid,
        z_grid,
        params_values_dict,
        sigma_obj_file=sigma_obj_file,
        skyfr_file=skyfr_file,
        filter_name=filter_name,
        theta_min=theta_min,
        theta_max=theta_max,
    )  # (n_z, n_m)

    # --- 4. Detection probability P_det(M, z) ---------------------------
    P_det = _pdet_grid(
        qbar_grid,
        sigma_lnY=sigma_lnY,
        q_cat=q_cat,
        n_grid=n_grid,
        nsig=nsig,
    )  # (n_z, n_m)

    # --- 5. Apply unresolved mask: keep *undetected* halos ---------------
    mask_mz = 1.0 - P_det  # (n_z, n_m)
    integrand = integrand * mask_mz[:, :, None]  # broadcast over ell

    # --- 6. Integrate over (lnM, z) for each ell ------------------------
    return _integrate_mz(integrand, z_grid, logm_grid)  # (n_ell,)


@partial(
    jax.jit,
    static_argnames=(
        "sigma_obj_file",
        "skyfr_file",
        "filter_name",
        "theta_min",
        "theta_max",
        "n_grid",
        "nsig",
    ),
)
def compute_masked_tsz_Dell(
    params_values_dict=None,
    *,
    q_cat,
    sigma_lnY,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
    n_grid=4096,
    nsig=16.0,
):
    r"""
    Compute D_ell for the masked (unresolved) tSZ map with full
    double-scatter completeness.

    .. math::

        D_\ell = \frac{\ell(\ell+1)}{2\pi}\,C_\ell^{yy,\mathrm{unres}}
                 \times 10^{12}

    Parameters
    ----------
    (same as ``compute_masked_tsz_ps``)

    Returns
    -------
    D_ell : (n_ell,) jnp.ndarray
        D_ell on the internal ell grid [dimensionless, scaled by 1e12].
    """
    ell = get_ell_range()  # internal ell grid

    C_ell = compute_masked_tsz_ps(
        params_values_dict=params_values_dict,
        q_cat=q_cat,
        sigma_lnY=sigma_lnY,
        sigma_obj_file=sigma_obj_file,
        skyfr_file=skyfr_file,
        filter_name=filter_name,
        theta_min=theta_min,
        theta_max=theta_max,
        n_grid=n_grid,
        nsig=nsig,
    )  # (n_ell,)

    D_ell = ell * (ell + 1.0) * C_ell / (2.0 * jnp.pi) * 1e12
    return D_ell


@partial(
    jax.jit,
    static_argnames=(
        "sigma_obj_file",
        "skyfr_file",
        "filter_name",
        "theta_min",
        "theta_max",
        "n_grid",
        "nsig",
        "scale_1e12",
    ),
)
def compute_masked_tsz_Dell_binned(
    params_values_dict=None,
    *,
    q_cat,
    sigma_lnY,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
    n_grid=4096,
    nsig=16.0,
    scale_1e12=True,
):
    r"""
    Compute **binned** D_ell (18 Planck-style bands) for the masked
    (unresolved) tSZ map with full double-scatter completeness.

    Binning procedure (same as ``bin_Dl_to_18``):
      1. Interpolate C_ell onto every integer ell within each
         [ELL_MIN, ELL_MAX] band.
      2. Convert to D_ell = ell(ell+1) C_ell / (2 pi) (* 1e12).
      3. Average D_ell uniformly over integer ell in each band.

    Parameters
    ----------
    (same as ``compute_masked_tsz_ps``, plus ``scale_1e12``)
    scale_1e12 : bool, default True
        If True, multiply D_ell by 1e12 (standard convention).

    Returns
    -------
    D_ell_binned : (18,) jnp.ndarray
        Binned D_ell values corresponding to the 18 effective multipoles.
    """
    ell = get_ell_range()

    C_ell = compute_masked_tsz_ps(
        params_values_dict=params_values_dict,
        q_cat=q_cat,
        sigma_lnY=sigma_lnY,
        sigma_obj_file=sigma_obj_file,
        skyfr_file=skyfr_file,
        filter_name=filter_name,
        theta_min=theta_min,
        theta_max=theta_max,
        n_grid=n_grid,
        nsig=nsig,
    )  # (n_ell,)

    # Bin C_ell -> D_ell in 18 bands (interpolate, convert, average)
    D_ell_binned = bin_Dl_to_18(ell, C_ell, scale_1e12=scale_1e12)  # (18,)
    return D_ell_binned

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
from .utils import get_ell_range, simpson, get_ell_binwidth
from .tsz import get_integral_grid, get_integral_grid_trisp
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


@partial(jax.jit, static_argnames=("n_power", "n_grid", "nsig"))
def _conditional_An_undetected_batch(
    qbar_flat, sigma_lnY, q_cat, *, n_power=2, n_grid=1024, nsig=8.0,
):
    r"""
    Conditional n-th moment  :math:`\langle A^n\,\mathbf{1}(q_{\rm obs}<q_{\rm cut})\rangle`
    for each (M, z) point — **no vmap**.

    When the Compton-y profile has lognormal intrinsic scatter
    :math:`y_\ell = A\,\tilde y_{\ell,0}` with
    :math:`\ln A \sim \mathcal{N}(0, \sigma_{\ln Y}^2)`,
    the masked 1-halo integrand that goes as :math:`A^n` must be weighted
    by this conditional moment rather than :math:`1 - P_{\rm det}`.

    Use ``n_power=2`` for the power spectrum (:math:`|y_\ell|^2`)
    and ``n_power=4`` for the trispectrum
    (:math:`|y_\ell|^2\,|y_{\ell'}|^2`).

    .. math::

        \langle A^n\,\mathbf{1}(q_{\rm obs} < q_{\rm cut})\rangle
        = \frac{1}{\sqrt{2\pi}}
          \int du\; e^{-u^2/2}\; e^{n\,\sigma_{\ln Y}\,u}\;
          \Phi\!\bigl(q_{\rm cut} - e^{\sigma_{\ln Y}\,u}\,\bar q_m\bigr)

    where :math:`u = \ln A / \sigma_{\ln Y}` is the standardised scatter
    variable and :math:`\Phi` is the standard-normal CDF.

    Parameters
    ----------
    qbar_flat : (N,) array
        Mean (theory) SNR for each (M, z) point.
    sigma_lnY : float
        Intrinsic log-normal scatter width.
    q_cat : float
        SNR detection threshold.
    n_power : int
        Power of A in the moment (2 for PS, 4 for trispectrum).
    n_grid : int
        Number of quadrature points.
    nsig : float
        Integration range in units of sigma_lnY.

    Returns
    -------
    cond_An : (N,) array
        Conditional n-th moment at each point.
    """
    import jax.scipy.special as jsp

    u = jnp.linspace(-nsig, nsig, n_grid)                  # (n_grid,)
    du = u[1] - u[0]
    gauss = jnp.exp(-0.5 * u * u)                          # (n_grid,)
    A_n = jnp.exp(n_power * sigma_lnY * u)                 # (n_grid,)

    mu = jnp.log(jnp.maximum(qbar_flat, 1e-30))            # (N,)
    q_m = jnp.exp(mu[:, None] + sigma_lnY * u[None, :])    # (N, n_grid)

    arg = (q_cat - q_m) / jnp.sqrt(2.0)                    # (N, n_grid)
    Phi = 0.5 * (1.0 + jsp.erf(arg))                       # (N, n_grid)

    integrand = gauss[None, :] * A_n[None, :] * Phi         # (N, n_grid)

    raw = jnp.trapezoid(integrand, dx=du, axis=1)           # (N,)
    return raw / jnp.sqrt(2.0 * jnp.pi)


@partial(jax.jit, static_argnames=("n_power", "n_grid", "nsig"))
def _conditional_An_undetected_grid(
    qbar_grid, *, sigma_lnY, q_cat, n_power=2, n_grid=1024, nsig=8.0,
):
    """
    Conditional n-th moment <A^n * 1(q_obs < q_cut)> on a (n_z, n_m) grid.

    n_power=2 for the power spectrum, n_power=4 for the trispectrum.
    """
    qbar_flat = qbar_grid.reshape(-1)
    result_flat = _conditional_An_undetected_batch(
        qbar_flat, sigma_lnY, q_cat,
        n_power=n_power, n_grid=n_grid, nsig=nsig,
    )
    return result_flat.reshape(qbar_grid.shape)


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
    n_grid=1024,
    nsig=18.0,
):
    r"""
    Compute the **unresolved** (masked) tSZ angular power spectrum C_ell
    using the conditional second moment for lognormal intrinsic scatter.

    Because the 1-halo tSZ power spectrum involves :math:`|\tilde y_\ell|^2`
    and the signal amplitude has lognormal scatter
    :math:`\ln A \sim \mathcal{N}(0,\,\sigma_{\ln Y}^2)`, the correct
    masked integrand uses the **conditional second moment** rather than
    a simple :math:`1 - P_{\rm det}` factor:

    .. math::

        C_\ell^{yy,\,\mathrm{unres}}
        = \int dz\;\frac{d^2V}{dz\,d\Omega}
          \int dM\;\frac{dn}{dM}\;
          |\tilde y_{\ell,0}(M,z)|^2\;
          \langle A^2\,\mathbf{1}(q_{\rm obs} < q_{\rm cut})\rangle

    where

    .. math::

        \langle A^2\,\mathbf{1}(q_{\rm obs} < q_{\rm cut})\rangle
        = \int d\ln A\;\mathcal{N}(\ln A;\,0,\,\sigma_{\ln Y}^2)\;
          A^2\;\Phi(q_{\rm cut} - A\,\bar q_m)

    and :math:`\Phi` is the standard-normal CDF.

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

    # --- 4. Conditional second moment <A^2 * 1(q_obs < q_cut)> ----------
    cond_A2 = _conditional_An_undetected_grid(
        qbar_grid,
        sigma_lnY=sigma_lnY,
        q_cat=q_cat,
        n_power=2,
        n_grid=n_grid,
        nsig=nsig,
    )  # (n_z, n_m)

    # --- 5. Weight integrand by conditional second moment ----------------
    integrand = integrand * cond_A2[:, :, None]  # broadcast over ell

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


# ---------------------------------------------------------------------------
#  Trispectrum with double-scatter completeness (GNFW)
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
def compute_masked_tsz_trispectrum(
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
    1-halo trispectrum T_{ell, ell'} for the **unresolved** (masked) tSZ map
    with lognormal intrinsic scatter in the y-profile amplitude.

    Because the trispectrum involves
    :math:`|y_\ell|^2\,|y_{\ell'}|^2 \propto A^4`,
    the correct masked integrand uses the **conditional fourth moment**:

    .. math::

        T_{\ell,\ell'}^{\rm unres}
        = \int dz \int d\ln M \;
          |\tilde y_{\ell,0}|^2\,|\tilde y_{\ell',0}|^2
          \;\frac{dn}{d\ln M}\;\frac{dV}{dz\,d\Omega}\;
          \langle A^4\,\mathbf{1}(q_{\rm obs} < q_{\rm cut})\rangle

    where

    .. math::

        \langle A^4\,\mathbf{1}(q_{\rm obs} < q_{\rm cut})\rangle
        = \int d\ln A\;\mathcal{N}(\ln A;\,0,\,\sigma_{\ln Y}^2)\;
          A^4\;\Phi(q_{\rm cut} - A\,\bar q_m)

    Parameters
    ----------
    params_values_dict : dict
        Cosmological + scaling-relation parameters.
    q_cat : float
        Catalogue SNR detection threshold.
    sigma_lnY : float
        Intrinsic log-normal scatter width.

    Returns
    -------
    ell_arr : (n_ell,) jnp.ndarray
    T_unres : (n_ell, n_ell) jnp.ndarray
    """
    allparams = classy_sz.get_all_relevant_params(
        params_values_dict=params_values_dict
    )
    z_grid = jnp.geomspace(allparams["z_min"], allparams["z_max"], 100)
    m_grid = jnp.geomspace(allparams["M_min"], allparams["M_max"], 100)
    logm_grid = jnp.log(m_grid)

    # Base trispectrum integrand (all halos, original GNFW amplitude)
    ell_arr, integrand = get_integral_grid_trisp(
        params_values_dict=params_values_dict
    )  # (n_z, n_m, n_ell, n_ell)

    # Mean SNR grid
    qbar_grid = build_snr_grid(
        m_grid, z_grid, params_values_dict,
        sigma_obj_file=sigma_obj_file,
        skyfr_file=skyfr_file,
        filter_name=filter_name,
        theta_min=theta_min,
        theta_max=theta_max,
    )  # (n_z, n_m)

    # Conditional fourth moment <A^4 * 1(q_obs < q_cut)>
    cond_A4 = _conditional_An_undetected_grid(
        qbar_grid,
        sigma_lnY=sigma_lnY,
        q_cat=q_cat,
        n_power=4,
        n_grid=n_grid,
        nsig=nsig,
    )  # (n_z, n_m)

    integrand = integrand * cond_A4[:, :, None, None]

    # Integrate over mass then redshift
    partial_m = simpson(integrand, x=logm_grid, axis=1)  # (n_z, n_ell, n_ell)
    T_unres = simpson(partial_m, x=z_grid, axis=0)        # (n_ell, n_ell)

    return ell_arr, T_unres


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
def compute_scaled_masked_tsz_trispectrum(
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
    Scaled masked trispectrum for D_ell = ell(ell+1) C_ell / (2pi),
    with double-scatter completeness.

    Returns
    -------
    ell_arr : (n_ell,)
    scaled_T : (n_ell, n_ell)
        T_{ll'} * [l(l+1) l'(l'+1)] / (2pi)^2 * 1e24
    """
    ell_arr, T_unres = compute_masked_tsz_trispectrum(
        params_values_dict=params_values_dict,
        q_cat=q_cat, sigma_lnY=sigma_lnY,
        sigma_obj_file=sigma_obj_file,
        skyfr_file=skyfr_file,
        filter_name=filter_name,
        theta_min=theta_min, theta_max=theta_max,
        n_grid=n_grid, nsig=nsig,
    )
    ell_factor = ell_arr * (ell_arr + 1.0)
    ell2D_factor = jnp.outer(ell_factor, ell_factor)
    prefactor = 1.0 / ((2.0 * jnp.pi) ** 2)
    return ell_arr, T_unres * ell2D_factor * prefactor * 1e24


# ---------------------------------------------------------------------------
#  Covariance with double-scatter completeness (GNFW)
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
def compute_masked_tsz_covariance(
    params_values_dict=None,
    *,
    q_cat,
    sigma_lnY,
    noise_ell=None,
    f_sky=1.0,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
    n_grid=4096,
    nsig=16.0,
):
    r"""
    Full covariance of the masked (unresolved) tSZ C_ell using
    double-scatter completeness.

    .. math::

        M_{\ell\ell'} = \frac{1}{4\pi f_{\rm sky}}
            \left[
                \frac{4\pi\,(C_\ell + N_\ell)^2}{(\ell+0.5)\,\Delta\ell}
                \;\delta_{\ell\ell'}
                + T_{\ell\ell'}^{\rm unres}
            \right]

    Returns
    -------
    ell_arr : (n_ell,)
    M       : (n_ell, n_ell)  full covariance (Gaussian + trispectrum)
    M_G     : (n_ell, n_ell)  Gaussian-only covariance
    """
    allparams = classy_sz.get_all_relevant_params(
        params_values_dict=params_values_dict
    )
    z_grid = jnp.geomspace(allparams["z_min"], allparams["z_max"], 100)
    m_grid = jnp.geomspace(allparams["M_min"], allparams["M_max"], 100)
    logm_grid = jnp.log(m_grid)
    ell_arr = get_ell_range()

    # Mean SNR grid
    qbar_grid = build_snr_grid(
        m_grid, z_grid, params_values_dict,
        sigma_obj_file=sigma_obj_file,
        skyfr_file=skyfr_file,
        filter_name=filter_name,
        theta_min=theta_min,
        theta_max=theta_max,
    )

    # Conditional second moment <A^2 * 1(undetected)> for power spectrum
    cond_A2 = _conditional_An_undetected_grid(
        qbar_grid,
        sigma_lnY=sigma_lnY,
        q_cat=q_cat,
        n_power=2,
        n_grid=n_grid,
        nsig=nsig,
    )  # (n_z, n_m)

    # Conditional fourth moment <A^4 * 1(undetected)> for trispectrum
    cond_A4 = _conditional_An_undetected_grid(
        qbar_grid,
        sigma_lnY=sigma_lnY,
        q_cat=q_cat,
        n_power=4,
        n_grid=n_grid,
        nsig=nsig,
    )  # (n_z, n_m)

    # --- Masked power spectrum C_ell (conditional second moment) ---
    integrand_C = get_integral_grid(
        params_values_dict=params_values_dict
    )  # (n_z, n_m, n_ell)
    integrand_C = integrand_C * cond_A2[:, :, None]
    C_yy_mask = _integrate_mz(integrand_C, z_grid, logm_grid)  # (n_ell,)

    # --- Masked trispectrum T_{ell,ell'} (conditional fourth moment) ---
    _, integrand_T = get_integral_grid_trisp(
        params_values_dict=params_values_dict
    )  # (n_z, n_m, n_ell, n_ell)
    integrand_T = integrand_T * cond_A4[:, :, None, None]

    partial_m = simpson(integrand_T, x=logm_grid, axis=1)
    T_mask = simpson(partial_m, x=z_grid, axis=0)  # (n_ell, n_ell)

    # --- Assemble covariance ---
    delta_ell = get_ell_binwidth()
    if noise_ell is None:
        noise_ell = jnp.zeros_like(C_yy_mask)

    diag_term = (4.0 * jnp.pi) * (C_yy_mask + noise_ell) ** 2 / (ell_arr + 0.5)
    M_G = jnp.diag(diag_term) / (4.0 * jnp.pi * f_sky * delta_ell)
    M = M_G + T_mask / (4.0 * jnp.pi * f_sky)

    return ell_arr, M, M_G


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
def compute_scaled_masked_tsz_covariance(
    params_values_dict=None,
    *,
    q_cat,
    sigma_lnY,
    noise_ell=None,
    f_sky=1.0,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
    n_grid=4096,
    nsig=16.0,
):
    r"""
    Scaled covariance for D_ell = ell(ell+1) C_ell / (2pi), on
    an SNR-masked (unresolved) tSZ map with double-scatter completeness.

    Returns
    -------
    ell_arr : (n_ell,)
    M_D     : (n_ell, n_ell)  covariance of D_ell  [*1e24]
    M_DG    : (n_ell, n_ell)  Gaussian-only         [*1e24]
    """
    ell_arr, M_C, M_CG = compute_masked_tsz_covariance(
        params_values_dict=params_values_dict,
        q_cat=q_cat, sigma_lnY=sigma_lnY,
        noise_ell=noise_ell, f_sky=f_sky,
        sigma_obj_file=sigma_obj_file,
        skyfr_file=skyfr_file,
        filter_name=filter_name,
        theta_min=theta_min, theta_max=theta_max,
        n_grid=n_grid, nsig=nsig,
    )

    s = ell_arr * (ell_arr + 1.0) / (2.0 * jnp.pi)
    S2D = jnp.outer(s, s)

    M_D  = S2D * M_C * 1e24
    M_DG = S2D * M_CG * 1e24

    return ell_arr, M_D, M_DG

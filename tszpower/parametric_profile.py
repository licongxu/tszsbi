"""
Parametric GNFW tSZ profile with variable amplitude.

This module replaces the fixed Arnaud et al. pressure-profile normalization
with a parametric amplitude model:

    y0(M, z) = 10^{A_SZ}  *  ((M/B) / (0.7 * 3e14))^{alpha_SZ}
                            *  E(z)^2  *  (h/0.7)^{-0.5}

where:
  - A_SZ     : log10 normalization (free parameter)
  - alpha_SZ : mass-scaling exponent (free parameter)
  - E(z) = H(z) / H0  (dimensionless Hubble rate)
  - h = H0 / 100
  - B is the hydrostatic mass bias factor

The GNFW *shape* (alpha, beta, gamma, c500) is unchanged; only the overall
amplitude is reparametrised.  Because y_ell(M, z, ell) is proportional to
y0(M, z), we rescale the existing pipeline via:

    ratio(M, z) = y0_parametric / y0_original

  - tSZ PS integrand scales as ratio^2  (goes as y_ell^2)
  - SNR scales as ratio                 (goes as y0 / sigma_noise)

Public API
----------
**Amplitude helpers:**
  - ``compute_y0_parametric``       : parametric y0
  - ``compute_snr_parametric``      : SNR with parametric y0

**tSZ power spectrum (full-sky, unmasked):**
  - ``compute_tsz_ps_parametric``          : C_ell
  - ``compute_tsz_Dell_parametric``        : D_ell
  - ``compute_tsz_Dell_binned_parametric`` : D_ell binned to 18 bands

**Cluster number counts (with full completeness):**
  - ``compute_number_detected_clusters_parametric`` : total N_det

**Masked tSZ power spectrum (with full completeness):**
  - ``compute_masked_tsz_ps_parametric``          : C_ell (unresolved)
  - ``compute_masked_tsz_Dell_parametric``        : D_ell (unresolved)
  - ``compute_masked_tsz_Dell_binned_parametric`` : D_ell binned to 18 bands
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

# --- intra-package imports --------------------------------------------------
from . import classy_sz
from .utils import get_ell_range, simpson, get_ell_binwidth
from .tsz import get_integral_grid, get_integral_grid_trisp, dVdzdOmega, get_integrand_number_counts
from .profiles import y_ell_interpolate
from .massfuncs import get_hmf_at_z_and_m
from .maskedpower import (
    compute_y0,
    compute_snr,
    compute_sigma_y0,
    compute_theta500_arcmin,
    build_snr_grid,
    completeness_convolution_jax,
    bin_Dl_to_18,
)


# ===================================================================
#  1. Amplitude helpers
# ===================================================================

def compute_y0_parametric(M, z, A_SZ, alpha_SZ, params_values_dict=None):
    r"""
    Parametric Compton-y amplitude:

    .. math::

        y_0 = 10^{A_{\rm SZ}}
              \left(\frac{M/B}{0.7 \times 3\times10^{14}}\right)^{\alpha_{\rm SZ}}
              E(z)^2 \; (h/0.7)^{-0.5}

    Parameters
    ----------
    M : float or jnp.ndarray
        Halo mass [M_sun].
    z : float or jnp.ndarray
        Redshift.
    A_SZ : float
        Log10 normalisation of the amplitude.
    alpha_SZ : float
        Mass-scaling exponent.
    params_values_dict : dict
        Cosmological + scaling-relation parameters.

    Returns
    -------
    y0 : same shape as M
    """
    rparams = classy_sz.get_all_relevant_params(
        params_values_dict=params_values_dict
    )
    conv_fac = 299792.458  # km/s
    h = rparams["H0"] / 100.0
    H_z = classy_sz.get_hubble_at_z(z, params_values_dict=params_values_dict) * conv_fac
    H0 = rparams["H0"]
    E_z = H_z / H0  # dimensionless

    B = rparams["B"]
    M_tilde = (M / B) / (0.7 * 3e14)

    y0 = (10.0 ** A_SZ) * (M_tilde ** alpha_SZ) * (E_z ** 2) * ((h / 0.7) ** (-0.5))
    return y0


def _y0_ratio(M, z, A_SZ, alpha_SZ, params_values_dict=None):
    """
    Rescaling factor:  y0_parametric / y0_original  for a single (M, z).
    Uses ratio = y0_param / y0_orig so that when parametric matches compute_y0
    (e.g. A_SZ from Arnaud fit), ratio ≈ 1 and parametric PS/CNC match the original.
    """
    y0_param = compute_y0_parametric(
        M, z, A_SZ, alpha_SZ, params_values_dict=params_values_dict
    )
    y0_orig = compute_y0(M, z, params_values_dict=params_values_dict)
    return y0_param / y0_orig


def _y0_ratio_grid(m_grid, z_grid, A_SZ, alpha_SZ, params_values_dict=None):
    """
    Build the ratio y0_param / y0_orig on a (n_z, n_m) grid.
    """
    def ratio_at_z(zz):
        return jax.vmap(
            lambda mm: _y0_ratio(mm, zz, A_SZ, alpha_SZ, params_values_dict)
        )(m_grid)

    return jax.vmap(ratio_at_z)(z_grid)  # (n_z, n_m)


def compute_snr_parametric(
    M, z, A_SZ, alpha_SZ, params_values_dict,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
):
    """
    SNR with the parametric amplitude:  SNR = y0_parametric / sigma_noise.

    sigma_noise depends only on theta_500 (angular size), which is
    independent of the signal amplitude, so:

        SNR_parametric = (y0_param / y0_orig) * SNR_original
    """
    ratio = _y0_ratio(M, z, A_SZ, alpha_SZ, params_values_dict)
    snr_orig = compute_snr(
        M, z, params_values_dict,
        sigma_obj_file=sigma_obj_file,
        skyfr_file=skyfr_file,
        filter_name=filter_name,
        theta_min=theta_min,
        theta_max=theta_max,
    )
    return snr_orig * ratio


def build_snr_grid_parametric(
    m_grid, z_grid, A_SZ, alpha_SZ, params_values_dict,
    *,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
):
    """
    Return snr_grid_parametric with shape (n_z, n_m).

    Uses the fast ratio approach:
        snr_param = ratio(M, z) * snr_original
    """
    snr_orig = build_snr_grid(
        m_grid, z_grid, params_values_dict,
        sigma_obj_file=sigma_obj_file,
        skyfr_file=skyfr_file,
        filter_name=filter_name,
        theta_min=theta_min,
        theta_max=theta_max,
    )  # (n_z, n_m)

    ratio_grid = _y0_ratio_grid(
        m_grid, z_grid, A_SZ, alpha_SZ, params_values_dict
    )  # (n_z, n_m)

    return snr_orig * ratio_grid


# ===================================================================
#  2. Internal integration helpers
# ===================================================================

@jax.jit
def _integrate_mz(integrand, z_grid, logm_grid):
    """
    Integrate (n_z, n_m, n_ell) integrand over mass then redshift
    for each multipole.  Returns C_ell of shape (n_ell,).
    """
    n_ell = integrand.shape[-1]

    def scan_body(_, i):
        integrand_i = integrand[:, :, i]
        partial_m = simpson(integrand_i, x=logm_grid, axis=1)
        result    = simpson(partial_m,   x=z_grid,    axis=0)
        return None, result

    _, C_yy = lax.scan(scan_body, None, jnp.arange(n_ell))
    return C_yy


@partial(jax.jit, static_argnames=("n_grid", "nsig"))
def _pdet_batch(qbar_flat, sigma_lnY, q_cat, *, n_grid=1024, nsig=8.0):
    r"""
    Batched P_det via double-scatter completeness — **no vmap**.

    Uses a standardised integration variable  u = (t − μ) / σ_lnY  ∈ [−nsig, nsig]
    so the Gaussian kernel  exp(−u²/2)  and the grid  u  are shared across
    all N input points.  Everything reduces to (N, n_grid) matrix operations.

    The σ_lnY Jacobian (dt = σ du) cancels with the normalisation
    2√(2π) σ, giving

        P_det = trapz(exp(−u²/2) · g, du) / (2√(2π))

    Parameters
    ----------
    qbar_flat : (N,) array
        Mean (theory) SNR for each (M, z) point.
    sigma_lnY : float
        Intrinsic log-normal scatter width.
    q_cat : float
        SNR detection threshold.
    n_grid : int
        Number of quadrature points.
    nsig : float
        Integration range in units of sigma_lnY.

    Returns
    -------
    P_det : (N,) array
        Detection probability at each point.
    """
    import jax.scipy.special as jsp

    # Fixed standardised grid (shared for all qbar)
    u = jnp.linspace(-nsig, nsig, n_grid)            # (n_grid,)
    du = u[1] - u[0]
    gauss = jnp.exp(-0.5 * u * u)                    # (n_grid,)

    # Physical SNR: q_m = qbar · exp(σ · u)
    mu = jnp.log(jnp.maximum(qbar_flat, 1e-30))      # (N,)
    q_m = jnp.exp(mu[:, None] + sigma_lnY * u[None, :])  # (N, n_grid)

    # Selection factor: P(detected | q_m)
    arg = (q_cat - q_m) / jnp.sqrt(2.0)              # (N, n_grid)
    g = 1.0 - jsp.erf(arg)                           # (N, n_grid)

    # Integrand = Gaussian kernel × selection
    integrand = gauss[None, :] * g                    # (N, n_grid)

    # Trapezoidal integration (uniform spacing du)
    raw = jnp.trapezoid(integrand, dx=du, axis=1)     # (N,)
    return raw / (2.0 * jnp.sqrt(2.0 * jnp.pi))


@partial(jax.jit, static_argnames=("n_grid", "nsig"))
def _pdet_grid(
    qbar_grid, *, sigma_lnY, q_cat, n_grid=1024, nsig=8.0,
):
    """
    Detection probability P_det(M, z) via double-scatter completeness.
    """
    qbar_flat = qbar_grid.reshape(-1)
    P_det_flat = _pdet_batch(qbar_flat, sigma_lnY, q_cat, n_grid=n_grid, nsig=nsig)
    return P_det_flat.reshape(qbar_grid.shape)


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


# ===================================================================
#  3. Full-sky (unmasked) tSZ power spectrum
# ===================================================================

@partial(
    jax.jit,
    static_argnames=(
        "sigma_obj_file", "skyfr_file", "filter_name",
        "theta_min", "theta_max",
    ),
)
def compute_tsz_ps_parametric(
    params_values_dict=None,
    *,
    A_SZ,
    alpha_SZ,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
):
    r"""
    Full-sky tSZ angular power spectrum C_ell with the parametric amplitude.

    The base integrand  y_ell^2 * dn/dlnM * dV/dz/dOmega  is rescaled by
    ratio(M, z)^2, where ratio = y0_param / y0_orig.

    Parameters
    ----------
    params_values_dict : dict
        Cosmological + scaling-relation parameters.
    A_SZ : float
        Log10 normalisation of the parametric amplitude.
    alpha_SZ : float
        Mass-scaling exponent.

    Returns
    -------
    C_yy : (n_ell,) jnp.ndarray
    """
    allparams = classy_sz.get_all_relevant_params(
        params_values_dict=params_values_dict
    )
    z_grid = jnp.geomspace(allparams["z_min"], allparams["z_max"], 100)
    m_grid = jnp.geomspace(allparams["M_min"], allparams["M_max"], 100)
    logm_grid = jnp.log(m_grid)

    # Base integrand (original amplitude)
    integrand = get_integral_grid(params_values_dict=params_values_dict)

    # Rescale by ratio^2
    ratio_grid = _y0_ratio_grid(
        m_grid, z_grid, A_SZ, alpha_SZ, params_values_dict
    )
    integrand = integrand * (ratio_grid[:, :, None] ** 2)

    return _integrate_mz(integrand, z_grid, logm_grid)


@partial(
    jax.jit,
    static_argnames=(
        "sigma_obj_file", "skyfr_file", "filter_name",
        "theta_min", "theta_max",
    ),
)
def compute_tsz_Dell_parametric(
    params_values_dict=None,
    *,
    A_SZ,
    alpha_SZ,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
):
    r"""
    D_ell = ell(ell+1) C_ell / (2 pi) * 1e12  with parametric amplitude.
    """
    ell = get_ell_range()
    C_ell = compute_tsz_ps_parametric(
        params_values_dict=params_values_dict,
        A_SZ=A_SZ, alpha_SZ=alpha_SZ,
    )
    return ell * (ell + 1.0) * C_ell / (2.0 * jnp.pi) * 1e12


@partial(
    jax.jit,
    static_argnames=(
        "sigma_obj_file", "skyfr_file", "filter_name",
        "theta_min", "theta_max", "scale_1e12",
    ),
)
def compute_tsz_Dell_binned_parametric(
    params_values_dict=None,
    *,
    A_SZ,
    alpha_SZ,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
    scale_1e12=True,
):
    r"""
    Binned D_ell (18 Planck-style bands) with parametric amplitude.
    """
    ell = get_ell_range()
    C_ell = compute_tsz_ps_parametric(
        params_values_dict=params_values_dict,
        A_SZ=A_SZ, alpha_SZ=alpha_SZ,
    )
    return bin_Dl_to_18(ell, C_ell, scale_1e12=scale_1e12)


# ===================================================================
#  4. Cluster number counts (with full double-scatter completeness)
# ===================================================================

@partial(
    jax.jit,
    static_argnames=(
        "sigma_obj_file", "skyfr_file", "filter_name",
        "theta_min", "theta_max", "n_grid", "nsig",
    ),
)
def compute_number_detected_clusters_parametric(
    params_values_dict=None,
    *,
    A_SZ,
    alpha_SZ,
    q_cat,
    sigma_lnY,
    f_sky=1.0,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
    n_grid=1024,
    nsig=8.0,
):
    r"""
    Number of detected clusters using the parametric amplitude and the
    full double-scatter completeness.

    .. math::

        N_{\rm det} = f_{\rm sky} \int dz \int d\ln M \;
                      \frac{dN}{dz\,d\ln M} \; P_{\rm det}(M,z)

    where q_bar(M,z) = SNR_parametric(M,z) is computed with the
    parametric y0.
    """
    allparams = classy_sz.get_all_relevant_params(
        params_values_dict=params_values_dict
    )
    z_grid = jnp.geomspace(allparams["z_min"], allparams["z_max"], 100)
    m_grid = jnp.geomspace(allparams["M_min"], allparams["M_max"], 100)
    logm_grid = jnp.log(m_grid)

    # Base number-count integrand: 4pi * dn/dlnM * dV/dz/dOmega
    integrand_full = get_integrand_number_counts(
        params_values_dict=params_values_dict
    )
    dN_dzdlnM = integrand_full[:, :, 0]  # (n_z, n_m)

    # Parametric SNR grid
    qbar_grid = build_snr_grid_parametric(
        m_grid, z_grid, A_SZ, alpha_SZ, params_values_dict,
        sigma_obj_file=sigma_obj_file,
        skyfr_file=skyfr_file,
        filter_name=filter_name,
        theta_min=theta_min,
        theta_max=theta_max,
    )

    # Detection probability
    P_det = _pdet_grid(
        qbar_grid, sigma_lnY=sigma_lnY, q_cat=q_cat,
        n_grid=n_grid, nsig=nsig,
    )

    # Integrate
    integrand = dN_dzdlnM * P_det
    partial_m = simpson(integrand, x=logm_grid, axis=1)
    N_fullsky = simpson(partial_m, x=z_grid, axis=0)

    return f_sky * N_fullsky


@partial(
    jax.jit,
    static_argnames=(
        "sigma_obj_file", "skyfr_file", "filter_name",
        "theta_min", "theta_max", "n_grid", "nsig",
    ),
)
def compute_cluster_counts_in_z_q_bins_parametric(
    params_values_dict=None,
    *,
    A_SZ,
    alpha_SZ,
    sigma_lnY,
    z_edges,
    q_edges,
    f_sky=1.0,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
    n_grid=1024,
    nsig=8.0,
):
    r"""
    Binned cluster counts N(z_i) and N(q_j) using the parametric amplitude
    and the full double-scatter completeness.

    Returns
    -------
    Nz : (n_z_bins,)
        Number of clusters in each redshift bin, marginalised over q >= q_min.
    Nq : (n_q_bins,)
        Number of clusters in each SNR bin, marginalised over z.
    """
    allparams = classy_sz.get_all_relevant_params(
        params_values_dict=params_values_dict
    )

    z_grid = jnp.geomspace(allparams["z_min"], allparams["z_max"], 100)
    m_grid = jnp.geomspace(allparams["M_min"], allparams["M_max"], 100)
    logm_grid = jnp.log(m_grid)

    # Build dN/(dz dlnM)
    def hmf_at_z(z):
        return get_hmf_at_z_and_m(z=z, m=m_grid, params_values_dict=params_values_dict)

    dndlnm = jax.vmap(hmf_at_z)(z_grid)
    dVdz = dVdzdOmega(z_grid, params_values_dict=params_values_dict)
    dN_dzdlnM = 4.0 * jnp.pi * dndlnm * dVdz[:, None]

    # Parametric SNR grid
    qbar_grid = build_snr_grid_parametric(
        m_grid, z_grid, A_SZ, alpha_SZ, params_values_dict,
        sigma_obj_file=sigma_obj_file,
        skyfr_file=skyfr_file,
        filter_name=filter_name,
        theta_min=theta_min,
        theta_max=theta_max,
    )
    qbar_flat = qbar_grid.reshape(-1)

    # Completeness helper (batched — no vmap)
    def Pdet(qc):
        return _pdet_batch(
            qbar_flat, sigma_lnY, qc, n_grid=n_grid, nsig=nsig
        ).reshape(qbar_grid.shape)

    # N(z): marginalise over q >= q_min
    q_min = q_edges[0]
    Pdet_qmin = Pdet(q_min)

    Nz = []
    for i in range(len(z_edges) - 1):
        zlo, zhi = z_edges[i], z_edges[i + 1]
        mask_z = (z_grid >= zlo) & (z_grid < zhi)
        Iz = simpson(
            dN_dzdlnM * Pdet_qmin * mask_z[:, None],
            x=logm_grid, axis=1,
        )
        Nz.append(f_sky * simpson(Iz, x=z_grid))
    Nz = jnp.asarray(Nz)

    # N(q): marginalise over all z, bin in q
    Nq = []
    for j in range(len(q_edges) - 1):
        Pdet_lo = Pdet(q_edges[j])
        Pdet_hi = Pdet(q_edges[j + 1])
        Pbin_q = jnp.clip(Pdet_lo - Pdet_hi, 0.0, 1.0)
        Iz = simpson(dN_dzdlnM * Pbin_q, x=logm_grid, axis=1)
        Nq.append(f_sky * simpson(Iz, x=z_grid))
    Nq = jnp.asarray(Nq)

    return Nz, Nq


@partial(
    jax.jit,
    static_argnames=(
        "sigma_obj_file", "skyfr_file", "filter_name",
        "theta_min", "theta_max", "n_grid", "nsig",
    ),
)
def compute_cluster_counts_2d_parametric(
    params_values_dict=None,
    *,
    A_SZ,
    alpha_SZ,
    sigma_lnY,
    z_edges,
    q_edges,
    f_sky=1.0,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
    n_grid=1024,
    nsig=8.0,
):
    r"""
    2D binned cluster counts N(z_i, q_j) using the parametric amplitude
    and the full double-scatter completeness.

    Returns
    -------
    N2d : (n_z_bins, n_q_bins)
    """
    allparams = classy_sz.get_all_relevant_params(
        params_values_dict=params_values_dict
    )

    z_grid = jnp.geomspace(allparams["z_min"], allparams["z_max"], 100)
    m_grid = jnp.geomspace(allparams["M_min"], allparams["M_max"], 100)
    logm_grid = jnp.log(m_grid)

    def hmf_at_z(z):
        return get_hmf_at_z_and_m(z=z, m=m_grid, params_values_dict=params_values_dict)

    dndlnm = jax.vmap(hmf_at_z)(z_grid)
    dVdz = dVdzdOmega(z_grid, params_values_dict=params_values_dict)
    dN_dzdlnM = 4.0 * jnp.pi * dndlnm * dVdz[:, None]

    # Parametric SNR grid
    qbar_grid = build_snr_grid_parametric(
        m_grid, z_grid, A_SZ, alpha_SZ, params_values_dict,
        sigma_obj_file=sigma_obj_file,
        skyfr_file=skyfr_file,
        filter_name=filter_name,
        theta_min=theta_min,
        theta_max=theta_max,
    )
    qbar_flat = qbar_grid.reshape(-1)

    # Completeness helper (batched — no vmap)
    def Pdet(qc):
        return _pdet_batch(
            qbar_flat, sigma_lnY, qc, n_grid=n_grid, nsig=nsig
        ).reshape(qbar_grid.shape)

    q_lo_all = q_edges[:-1]
    q_hi_all = q_edges[1:]

    def Pbin_from_edges(q_lo, q_hi):
        return jnp.clip(Pdet(q_lo) - Pdet(q_hi), 0.0, 1.0)

    Pbin_all = jax.vmap(Pbin_from_edges)(q_lo_all, q_hi_all)

    zlo_all = z_edges[:-1]
    zhi_all = z_edges[1:]

    def count_one_z_bin(zlo, zhi, Pbin_q):
        mask_z = (z_grid >= zlo) & (z_grid < zhi)
        Iz = simpson(
            dN_dzdlnM * Pbin_q * mask_z[:, None],
            x=logm_grid, axis=1,
        )
        return f_sky * simpson(Iz, x=z_grid)

    def count_one_q_bin(Pbin_q):
        return jax.vmap(
            lambda zlo, zhi: count_one_z_bin(zlo, zhi, Pbin_q)
        )(zlo_all, zhi_all)

    N_qz = jax.vmap(count_one_q_bin)(Pbin_all)
    N2d = jnp.swapaxes(N_qz, 0, 1)

    return N2d


# ===================================================================
#  5. Masked (unresolved) tSZ power spectrum (with full completeness)
# ===================================================================

@partial(
    jax.jit,
    static_argnames=(
        "sigma_obj_file", "skyfr_file", "filter_name",
        "theta_min", "theta_max", "n_grid", "nsig",
    ),
)
def compute_masked_tsz_ps_parametric(
    params_values_dict=None,
    *,
    A_SZ,
    alpha_SZ,
    q_cat,
    sigma_lnY,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
    n_grid=1024,
    nsig=8.0,
):
    r"""
    Unresolved (masked) tSZ C_ell with parametric amplitude and
    lognormal intrinsic scatter in the y-profile amplitude.

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
    A_SZ : float
        Log10 normalisation.
    alpha_SZ : float
        Mass-scaling exponent.
    q_cat : float
        Catalogue SNR detection threshold.
    sigma_lnY : float
        Intrinsic log-normal scatter width.

    Returns
    -------
    C_yy_unres : (n_ell,) jnp.ndarray
    """
    allparams = classy_sz.get_all_relevant_params(
        params_values_dict=params_values_dict
    )
    z_grid = jnp.geomspace(allparams["z_min"], allparams["z_max"], 100)
    m_grid = jnp.geomspace(allparams["M_min"], allparams["M_max"], 100)
    logm_grid = jnp.log(m_grid)

    # --- Base integrand rescaled by ratio^2 ---
    integrand = get_integral_grid(params_values_dict=params_values_dict)

    ratio_grid = _y0_ratio_grid(
        m_grid, z_grid, A_SZ, alpha_SZ, params_values_dict
    )
    integrand = integrand * (ratio_grid[:, :, None] ** 2)

    # --- Parametric SNR grid ---
    qbar_grid = build_snr_grid_parametric(
        m_grid, z_grid, A_SZ, alpha_SZ, params_values_dict,
        sigma_obj_file=sigma_obj_file,
        skyfr_file=skyfr_file,
        filter_name=filter_name,
        theta_min=theta_min,
        theta_max=theta_max,
    )

    # --- Conditional second moment <A^2 * 1(q_obs < q_cut)> ---
    cond_A2 = _conditional_An_undetected_grid(
        qbar_grid, sigma_lnY=sigma_lnY, q_cat=q_cat,
        n_power=2, n_grid=n_grid, nsig=nsig,
    )

    integrand = integrand * cond_A2[:, :, None]

    return _integrate_mz(integrand, z_grid, logm_grid)


@partial(
    jax.jit,
    static_argnames=(
        "sigma_obj_file", "skyfr_file", "filter_name",
        "theta_min", "theta_max", "n_grid", "nsig",
    ),
)
def compute_masked_tsz_Dell_parametric(
    params_values_dict=None,
    *,
    A_SZ,
    alpha_SZ,
    q_cat,
    sigma_lnY,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
    n_grid=1024,
    nsig=8.0,
):
    r"""
    D_ell for the masked tSZ map with parametric amplitude.

    .. math::
        D_\ell = \frac{\ell(\ell+1)}{2\pi} C_\ell^{\rm unres} \times 10^{12}
    """
    ell = get_ell_range()
    C_ell = compute_masked_tsz_ps_parametric(
        params_values_dict=params_values_dict,
        A_SZ=A_SZ, alpha_SZ=alpha_SZ,
        q_cat=q_cat, sigma_lnY=sigma_lnY,
        sigma_obj_file=sigma_obj_file,
        skyfr_file=skyfr_file,
        filter_name=filter_name,
        theta_min=theta_min, theta_max=theta_max,
        n_grid=n_grid, nsig=nsig,
    )
    return ell * (ell + 1.0) * C_ell / (2.0 * jnp.pi) * 1e12


@partial(
    jax.jit,
    static_argnames=(
        "sigma_obj_file", "skyfr_file", "filter_name",
        "theta_min", "theta_max", "n_grid", "nsig",
        "scale_1e12",
    ),
)
def compute_masked_tsz_Dell_binned_parametric(
    params_values_dict=None,
    *,
    A_SZ,
    alpha_SZ,
    q_cat,
    sigma_lnY,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
    n_grid=1024,
    nsig=8.0,
    scale_1e12=True,
):
    r"""
    Binned D_ell (18 Planck-style bands) for the masked tSZ map with
    parametric amplitude and full double-scatter completeness.
    """
    ell = get_ell_range()
    C_ell = compute_masked_tsz_ps_parametric(
        params_values_dict=params_values_dict,
        A_SZ=A_SZ, alpha_SZ=alpha_SZ,
        q_cat=q_cat, sigma_lnY=sigma_lnY,
        sigma_obj_file=sigma_obj_file,
        skyfr_file=skyfr_file,
        filter_name=filter_name,
        theta_min=theta_min, theta_max=theta_max,
        n_grid=n_grid, nsig=nsig,
    )
    return bin_Dl_to_18(ell, C_ell, scale_1e12=scale_1e12)


# ===================================================================
#  6. Full-sky trispectrum (parametric amplitude)
# ===================================================================

@partial(
    jax.jit,
    static_argnames=(
        "sigma_obj_file", "skyfr_file", "filter_name",
        "theta_min", "theta_max",
    ),
)
def compute_trispectrum_parametric(
    params_values_dict=None,
    *,
    A_SZ,
    alpha_SZ,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
):
    r"""
    Full-sky 1-halo tSZ trispectrum T_{ell, ell'} with parametric amplitude.

    The base integrand  y_ell^2 * y_ell'^2 * dn/dlnM * dV/dz/dOmega
    is rescaled by ratio(M,z)^4, where ratio = y0_param / y0_orig.

    Parameters
    ----------
    params_values_dict : dict
        Cosmological + scaling-relation parameters.
    A_SZ : float
        Log10 normalisation of the parametric amplitude.
    alpha_SZ : float
        Mass-scaling exponent.

    Returns
    -------
    ell : (n_ell,) jnp.ndarray
    T_ell_ellprime : (n_ell, n_ell) jnp.ndarray
    """
    allparams = classy_sz.get_all_relevant_params(
        params_values_dict=params_values_dict
    )
    z_grid = jnp.geomspace(allparams["z_min"], allparams["z_max"], 100)
    m_grid = jnp.geomspace(allparams["M_min"], allparams["M_max"], 100)
    logm_grid = jnp.log(m_grid)

    # Build base trispectrum integrand (original amplitude)
    ell, integrand = get_integral_grid_trisp(params_values_dict=params_values_dict)
    # integrand shape: (n_z, n_m, n_ell, n_ell)

    # Rescale by ratio^4  (trispectrum goes as y0^4)
    ratio_grid = _y0_ratio_grid(
        m_grid, z_grid, A_SZ, alpha_SZ, params_values_dict
    )  # (n_z, n_m)
    integrand = integrand * (ratio_grid[:, :, None, None] ** 4)

    # Integrate over mass then redshift
    partial_m = simpson(integrand, x=logm_grid, axis=1)  # (n_z, n_ell, n_ell)
    T_ell_ellprime = simpson(partial_m, x=z_grid, axis=0)  # (n_ell, n_ell)

    return ell, T_ell_ellprime


@partial(
    jax.jit,
    static_argnames=(
        "sigma_obj_file", "skyfr_file", "filter_name",
        "theta_min", "theta_max",
    ),
)
def compute_scaled_trispectrum_parametric(
    params_values_dict=None,
    *,
    A_SZ,
    alpha_SZ,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
):
    r"""
    Scaled full-sky trispectrum for D_ell = ell(ell+1) C_ell / (2pi).

    Returns
    -------
    ell : (n_ell,)
    scaled_T : (n_ell, n_ell)
        scaled_T_{ll'} = T_{ll'} * [l(l+1) l'(l'+1)] / (2pi)^2 * 1e24
    """
    ell, T_llp = compute_trispectrum_parametric(
        params_values_dict=params_values_dict,
        A_SZ=A_SZ, alpha_SZ=alpha_SZ,
    )
    ell_factor = ell * (ell + 1.0)
    ell2D_factor = jnp.outer(ell_factor, ell_factor)
    prefactor = 1.0 / ((2.0 * jnp.pi) ** 2)
    return ell, T_llp * ell2D_factor * prefactor * 1e24


# ===================================================================
#  7. Full-sky covariance (parametric amplitude)
# ===================================================================

@partial(
    jax.jit,
    static_argnames=(
        "sigma_obj_file", "skyfr_file", "filter_name",
        "theta_min", "theta_max",
    ),
)
def compute_tsz_covariance_parametric(
    params_values_dict=None,
    *,
    A_SZ,
    alpha_SZ,
    noise_ell=None,
    f_sky=1.0,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
):
    r"""
    Full covariance of the full-sky tSZ C_ell with parametric amplitude.

    M_{ll'} = diag[ 4pi (C_ell + N_ell)^2 / (ell + 0.5) ] / (4pi f_sky delta_ell)
              + T_{ell,ell'} / (4pi f_sky)

    where C_ell and T_{ell,ell'} are computed with the parametric amplitude.

    Returns
    -------
    ell_arr : (n_ell,)
    M       : (n_ell, n_ell)  full covariance (Gaussian + trispectrum)
    M_G     : (n_ell, n_ell)  Gaussian-only covariance
    """
    # Parametric power spectrum
    C_yy = compute_tsz_ps_parametric(
        params_values_dict=params_values_dict,
        A_SZ=A_SZ, alpha_SZ=alpha_SZ,
    )  # (n_ell,)

    # Parametric trispectrum
    ell_arr, T_llp = compute_trispectrum_parametric(
        params_values_dict=params_values_dict,
        A_SZ=A_SZ, alpha_SZ=alpha_SZ,
    )

    delta_ell = get_ell_binwidth()

    if noise_ell is None:
        noise_ell = jnp.zeros_like(C_yy)

    diag_term = (4.0 * jnp.pi) * (C_yy + noise_ell) ** 2 / (ell_arr + 0.5)

    M_G = jnp.diag(diag_term) / (4.0 * jnp.pi * f_sky * delta_ell)
    M = M_G + T_llp / (4.0 * jnp.pi * f_sky)

    return ell_arr, M, M_G


# ===================================================================
#  8. Masked trispectrum (parametric amplitude + SNR-based mask)
# ===================================================================

def compute_trispectrum_masked_parametric(
    params_values_dict=None,
    *,
    A_SZ,
    alpha_SZ,
    q_obs=None,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
):
    r"""
    1-halo trispectrum for an SNR-masked (unresolved) tSZ map
    with parametric amplitude.

    Mask convention:
        W(M,z;q) = Theta(q - SNR_param(M,z))  -> keeps UNRESOLVED objects

    The integrand y_ell^2 * y_ell'^2 is rescaled by ratio(M,z)^4
    and the SNR entering the mask is the parametric SNR.

    Parameters
    ----------
    A_SZ : float
        Log10 normalisation.
    alpha_SZ : float
        Mass-scaling exponent.
    q_obs : float or None
        SNR threshold for masking.  None => no masking (full-sky).

    Returns
    -------
    ell_arr : (n_ell,) jnp.ndarray
    T_mask  : (n_ell, n_ell) jnp.ndarray   trispectrum only (no Gaussian term)
    """
    allparams = classy_sz.get_all_relevant_params(
        params_values_dict=params_values_dict
    )
    z_grid = jnp.geomspace(allparams["z_min"], allparams["z_max"], 100)
    m_grid = jnp.geomspace(allparams["M_min"], allparams["M_max"], 100)
    logm_grid = jnp.log(m_grid)
    ell_arr = get_ell_range()

    # --- Parametric SNR grid and Heaviside mask ---
    snr_grid = build_snr_grid_parametric(
        m_grid, z_grid, A_SZ, alpha_SZ, params_values_dict,
        sigma_obj_file=sigma_obj_file,
        skyfr_file=skyfr_file,
        filter_name=filter_name,
        theta_min=theta_min,
        theta_max=theta_max,
    )  # (n_z, n_m)

    if q_obs is None:
        mask_mz = jnp.ones_like(snr_grid, dtype=float)
    else:
        q_obs = jnp.asarray(q_obs, dtype=snr_grid.dtype)
        mask_mz = jnp.heaviside(q_obs - snr_grid, 0.0)

    # --- Build y_ell grid ---
    def get_yellm_for_z(zp):
        _, y_ellm = y_ell_interpolate(zp, m_grid, params_values_dict=params_values_dict)
        return y_ellm  # (n_m, n_ell)

    y_ell_mz = jax.vmap(get_yellm_for_z)(z_grid)  # (n_z, n_m, n_ell)
    y_sq = y_ell_mz ** 2  # y_ell^2 with original amplitude

    # --- HMF and comoving volume ---
    def get_hmf_for_z(zp):
        return get_hmf_at_z_and_m(z=zp, m=m_grid, params_values_dict=params_values_dict)

    dndlnm = jax.vmap(get_hmf_for_z)(z_grid)  # (n_z, n_m)
    comov = dVdzdOmega(z_grid, params_values_dict=params_values_dict)  # (n_z,)

    # --- Ratio^4 rescaling for parametric amplitude ---
    ratio_grid = _y0_ratio_grid(
        m_grid, z_grid, A_SZ, alpha_SZ, params_values_dict
    )  # (n_z, n_m)

    # --- Trispectrum integrand ---
    integrand_T = (
        y_sq[:, :, :, None] * y_sq[:, :, None, :]
        * (ratio_grid[:, :, None, None] ** 4)
        * dndlnm[:, :, None, None]
        * comov[:, None, None, None]
        * mask_mz[:, :, None, None]
    )  # (n_z, n_m, n_ell, n_ell)

    partial_m = simpson(integrand_T, x=logm_grid, axis=1)  # (n_z, n_ell, n_ell)
    T_mask = simpson(partial_m, x=z_grid, axis=0)           # (n_ell, n_ell)

    return ell_arr, T_mask


def compute_scaled_trispectrum_masked_parametric(
    params_values_dict=None,
    *,
    A_SZ,
    alpha_SZ,
    q_obs=None,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
):
    r"""
    Scaled masked trispectrum for D_ell = ell(ell+1) C_ell / (2pi),
    with parametric amplitude.

    Returns
    -------
    ell_arr : (n_ell,)
    scaled_T : (n_ell, n_ell)
        scaled_T_{ll'} = T_{ll'} * [l(l+1) l'(l'+1)] / (2pi)^2 * 1e24
    """
    ell_arr, T_mask = compute_trispectrum_masked_parametric(
        params_values_dict=params_values_dict,
        A_SZ=A_SZ, alpha_SZ=alpha_SZ,
        q_obs=q_obs,
        sigma_obj_file=sigma_obj_file,
        skyfr_file=skyfr_file,
        filter_name=filter_name,
        theta_min=theta_min,
        theta_max=theta_max,
    )
    ell_factor = ell_arr * (ell_arr + 1.0)
    ell2D_factor = jnp.outer(ell_factor, ell_factor)
    prefactor = 1.0 / ((2.0 * jnp.pi) ** 2)
    return ell_arr, T_mask * ell2D_factor * prefactor * 1e24


# ===================================================================
#  9. Masked covariance (parametric amplitude + SNR-based mask)
# ===================================================================

def compute_tsz_covariance_masked_parametric(
    params_values_dict=None,
    *,
    A_SZ,
    alpha_SZ,
    q_obs=None,
    noise_ell=None,
    f_sky=1.0,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
):
    r"""
    Covariance matrix for an SNR-masked (unresolved) tSZ map,
    with parametric amplitude.  Includes 1-halo trispectrum.

    M_{ll'} = diag[ 4pi (C_ell + N_ell)^2 / (ell + 0.5) ] / (4pi f_sky delta_ell)
              + T_{ell,ell'} / (4pi f_sky)

    Both C_ell and T_{ell,ell'} use the parametric amplitude and
    the parametric-SNR Heaviside mask.

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

    # --- Parametric SNR grid and Heaviside mask ---
    snr_grid = build_snr_grid_parametric(
        m_grid, z_grid, A_SZ, alpha_SZ, params_values_dict,
        sigma_obj_file=sigma_obj_file,
        skyfr_file=skyfr_file,
        filter_name=filter_name,
        theta_min=theta_min,
        theta_max=theta_max,
    )  # (n_z, n_m)

    if q_obs is None:
        mask_mz = jnp.ones_like(snr_grid, dtype=float)
    else:
        q_obs_arr = jnp.asarray(q_obs, dtype=snr_grid.dtype)
        mask_mz = jnp.heaviside(q_obs_arr - snr_grid, 0.0)

    # --- Build y_ell grid ---
    def get_yellm_for_z(zp):
        _, y_ellm = y_ell_interpolate(zp, m_grid, params_values_dict=params_values_dict)
        return y_ellm

    y_ell_mz = jax.vmap(get_yellm_for_z)(z_grid)  # (n_z, n_m, n_ell)
    y_sq = y_ell_mz ** 2

    # --- HMF and comoving volume ---
    def get_hmf_for_z(zp):
        return get_hmf_at_z_and_m(z=zp, m=m_grid, params_values_dict=params_values_dict)

    dndlnm = jax.vmap(get_hmf_for_z)(z_grid)
    comov = dVdzdOmega(z_grid, params_values_dict=params_values_dict)

    # --- Ratio grid for parametric amplitude ---
    ratio_grid = _y0_ratio_grid(
        m_grid, z_grid, A_SZ, alpha_SZ, params_values_dict
    )  # (n_z, n_m)

    # ------------------------------------------------------------------
    # Masked power spectrum C_ell: ratio^2 * integrand * mask
    # ------------------------------------------------------------------
    integrand_C = get_integral_grid(params_values_dict=params_values_dict)  # (n_z, n_m, n_ell)
    integrand_C = integrand_C * (ratio_grid[:, :, None] ** 2) * mask_mz[:, :, None]

    C_yy_mask = _integrate_mz(integrand_C, z_grid, logm_grid)  # (n_ell,)

    # ------------------------------------------------------------------
    # Masked trispectrum T_{ell,ell'}: ratio^4 * integrand * mask
    # ------------------------------------------------------------------
    integrand_T = (
        y_sq[:, :, :, None] * y_sq[:, :, None, :]
        * (ratio_grid[:, :, None, None] ** 4)
        * dndlnm[:, :, None, None]
        * comov[:, None, None, None]
        * mask_mz[:, :, None, None]
    )  # (n_z, n_m, n_ell, n_ell)

    partial_m = simpson(integrand_T, x=logm_grid, axis=1)
    T_mask = simpson(partial_m, x=z_grid, axis=0)  # (n_ell, n_ell)

    # ------------------------------------------------------------------
    # Assemble covariance
    # ------------------------------------------------------------------
    delta_ell = get_ell_binwidth()
    if noise_ell is None:
        noise_ell = jnp.zeros_like(C_yy_mask)

    diag_term = (4.0 * jnp.pi) * (C_yy_mask + noise_ell) ** 2 / (ell_arr + 0.5)

    M_G = jnp.diag(diag_term) / (4.0 * jnp.pi * f_sky * delta_ell)
    M = M_G + T_mask / (4.0 * jnp.pi * f_sky)

    return ell_arr, M, M_G


def compute_scaled_tsz_covariance_masked_parametric(
    params_values_dict=None,
    *,
    A_SZ,
    alpha_SZ,
    q_obs=None,
    noise_ell=None,
    f_sky=1.0,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
):
    r"""
    Scaled covariance for D_ell = ell(ell+1) C_ell / (2pi), on an
    SNR-masked (unresolved) tSZ map with parametric amplitude.

    Returns
    -------
    ell_arr : (n_ell,)
    M_D     : (n_ell, n_ell)  covariance of D_ell (Gaussian + trispectrum)  [*1e24]
    M_DG    : (n_ell, n_ell)  Gaussian-only covariance of D_ell              [*1e24]
    """
    ell_arr, M_C, M_CG = compute_tsz_covariance_masked_parametric(
        params_values_dict=params_values_dict,
        A_SZ=A_SZ, alpha_SZ=alpha_SZ,
        q_obs=q_obs,
        noise_ell=noise_ell,
        f_sky=f_sky,
        sigma_obj_file=sigma_obj_file,
        skyfr_file=skyfr_file,
        filter_name=filter_name,
        theta_min=theta_min,
        theta_max=theta_max,
    )

    # D_ell = s_ell * C_ell, s_ell = ell(ell+1) / (2 pi)
    s = ell_arr * (ell_arr + 1.0) / (2.0 * jnp.pi)
    S2D = jnp.outer(s, s)

    M_D  = S2D * M_C * 1e24
    M_DG = S2D * M_CG * 1e24

    return ell_arr, M_D, M_DG


# ===================================================================
#  10. Masked trispectrum with double-scatter completeness (parametric)
# ===================================================================

@partial(
    jax.jit,
    static_argnames=(
        "sigma_obj_file", "skyfr_file", "filter_name",
        "theta_min", "theta_max", "n_grid", "nsig",
    ),
)
def compute_masked_tsz_trispectrum_parametric(
    params_values_dict=None,
    *,
    A_SZ,
    alpha_SZ,
    q_cat,
    sigma_lnY,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
    n_grid=1024,
    nsig=8.0,
):
    r"""
    1-halo trispectrum for the **unresolved** tSZ map with parametric
    amplitude and lognormal intrinsic scatter.

    Because the trispectrum involves
    :math:`|y_\ell|^2\,|y_{\ell'}|^2 \propto A^4`,
    the correct masked integrand uses the **conditional fourth moment**:

    .. math::

        T_{\ell,\ell'}^{\rm unres}
        = \int dz \int d\ln M \;
          \mathrm{ratio}^4\;
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
    A_SZ : float
        Log10 normalisation.
    alpha_SZ : float
        Mass-scaling exponent.
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

    # Base trispectrum integrand (original GNFW amplitude)
    ell_arr, integrand = get_integral_grid_trisp(
        params_values_dict=params_values_dict
    )  # (n_z, n_m, n_ell, n_ell)

    # Ratio^4 rescaling for parametric amplitude
    ratio_grid = _y0_ratio_grid(
        m_grid, z_grid, A_SZ, alpha_SZ, params_values_dict
    )  # (n_z, n_m)
    integrand = integrand * (ratio_grid[:, :, None, None] ** 4)

    # Parametric SNR grid
    qbar_grid = build_snr_grid_parametric(
        m_grid, z_grid, A_SZ, alpha_SZ, params_values_dict,
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
        "sigma_obj_file", "skyfr_file", "filter_name",
        "theta_min", "theta_max", "n_grid", "nsig",
    ),
)
def compute_scaled_masked_tsz_trispectrum_parametric(
    params_values_dict=None,
    *,
    A_SZ,
    alpha_SZ,
    q_cat,
    sigma_lnY,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
    n_grid=1024,
    nsig=8.0,
):
    r"""
    Scaled masked trispectrum for D_ell with parametric amplitude
    and double-scatter completeness.

    Returns
    -------
    ell_arr : (n_ell,)
    scaled_T : (n_ell, n_ell)
        T_{ll'} * [l(l+1) l'(l'+1)] / (2pi)^2 * 1e24
    """
    ell_arr, T_unres = compute_masked_tsz_trispectrum_parametric(
        params_values_dict=params_values_dict,
        A_SZ=A_SZ, alpha_SZ=alpha_SZ,
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


# ===================================================================
#  11. Masked covariance with double-scatter completeness (parametric)
# ===================================================================

@partial(
    jax.jit,
    static_argnames=(
        "sigma_obj_file", "skyfr_file", "filter_name",
        "theta_min", "theta_max", "n_grid", "nsig",
    ),
)
def compute_masked_tsz_covariance_parametric(
    params_values_dict=None,
    *,
    A_SZ,
    alpha_SZ,
    q_cat,
    sigma_lnY,
    noise_ell=None,
    f_sky=1.0,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
    n_grid=1024,
    nsig=8.0,
):
    r"""
    Covariance matrix for the masked (unresolved) tSZ map with
    parametric amplitude and full double-scatter completeness.

    Both the masked C_ell (Gaussian diagonal) and T_{ell,ell'}
    (non-Gaussian off-diagonal) use ratio rescaling and completeness.

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

    # --- Ratio grid for parametric amplitude ---
    ratio_grid = _y0_ratio_grid(
        m_grid, z_grid, A_SZ, alpha_SZ, params_values_dict
    )  # (n_z, n_m)

    # --- Parametric SNR -> double-scatter completeness ---
    qbar_grid = build_snr_grid_parametric(
        m_grid, z_grid, A_SZ, alpha_SZ, params_values_dict,
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

    # --- Masked power spectrum C_ell: ratio^2 * integrand * <A^2 * 1(undet)> ---
    integrand_C = get_integral_grid(
        params_values_dict=params_values_dict
    )  # (n_z, n_m, n_ell)
    integrand_C = integrand_C * (ratio_grid[:, :, None] ** 2) * cond_A2[:, :, None]
    C_yy_mask = _integrate_mz(integrand_C, z_grid, logm_grid)  # (n_ell,)

    # --- Masked trispectrum T_{ell,ell'}: ratio^4 * integrand * <A^4 * 1(undet)> ---
    _, integrand_T = get_integral_grid_trisp(
        params_values_dict=params_values_dict
    )  # (n_z, n_m, n_ell, n_ell)
    integrand_T = (
        integrand_T
        * (ratio_grid[:, :, None, None] ** 4)
        * cond_A4[:, :, None, None]
    )

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
        "sigma_obj_file", "skyfr_file", "filter_name",
        "theta_min", "theta_max", "n_grid", "nsig",
    ),
)
def compute_scaled_masked_tsz_covariance_parametric(
    params_values_dict=None,
    *,
    A_SZ,
    alpha_SZ,
    q_cat,
    sigma_lnY,
    noise_ell=None,
    f_sky=1.0,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
    n_grid=1024,
    nsig=8.0,
):
    r"""
    Scaled covariance for D_ell on the masked tSZ map with parametric
    amplitude and double-scatter completeness.

    Returns
    -------
    ell_arr : (n_ell,)
    M_D     : (n_ell, n_ell)  covariance of D_ell  [*1e24]
    M_DG    : (n_ell, n_ell)  Gaussian-only         [*1e24]
    """
    ell_arr, M_C, M_CG = compute_masked_tsz_covariance_parametric(
        params_values_dict=params_values_dict,
        A_SZ=A_SZ, alpha_SZ=alpha_SZ,
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

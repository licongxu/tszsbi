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
from .utils import get_ell_range, simpson
from .tsz import get_integral_grid, dVdzdOmega, get_integrand_number_counts
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
def _pdet_grid(
    qbar_grid, *, sigma_lnY, q_cat, n_grid=4096, nsig=16.0,
):
    """
    Detection probability P_det(M, z) via double-scatter completeness.
    """
    qbar_flat = qbar_grid.reshape(-1)
    raw = jax.vmap(
        lambda qb: completeness_convolution_jax(
            qb, sigma_lnY, q_cat, n_grid=n_grid, nsig=nsig
        )
    )(qbar_flat)
    norm = 2.0 * jnp.sqrt(2.0 * jnp.pi) * sigma_lnY
    return (raw / norm).reshape(qbar_grid.shape)


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
    n_grid=4096,
    nsig=16.0,
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
    n_grid=4096,
    nsig=16.0,
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
    n_z_hmf = int(allparams.get("n_z", 1024))
    n_m_hmf = int(allparams.get("n_points_data_lik", 100))

    z_grid = jnp.geomspace(allparams["z_min"], allparams["z_max"], n_z_hmf)
    m_grid = jnp.geomspace(allparams["M_min"], allparams["M_max"], n_m_hmf)
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

    # Completeness helper
    norm = 2.0 * jnp.sqrt(2.0 * jnp.pi) * sigma_lnY

    def Pdet(qc):
        raw = jax.vmap(
            lambda qb: completeness_convolution_jax(
                qb, sigma_lnY, qc, n_grid=n_grid, nsig=nsig
            )
        )(qbar_flat)
        return (raw / norm).reshape(qbar_grid.shape)

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
    n_grid=4096,
    nsig=16.0,
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
    n_z_hmf = int(allparams.get("n_z", 1024))
    n_m_hmf = int(allparams.get("n_points_data_lik", 50))

    z_grid = jnp.geomspace(allparams["z_min"], allparams["z_max"], n_z_hmf)
    m_grid = jnp.geomspace(allparams["M_min"], allparams["M_max"], n_m_hmf)
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

    norm = 2.0 * jnp.sqrt(2.0 * jnp.pi) * sigma_lnY

    def Pdet(qc):
        raw = jax.vmap(
            lambda qb: completeness_convolution_jax(
                qb, sigma_lnY, qc, n_grid=n_grid, nsig=nsig
            )
        )(qbar_flat)
        return (raw / norm).reshape(qbar_grid.shape)

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
    n_grid=4096,
    nsig=16.0,
):
    r"""
    Unresolved (masked) tSZ C_ell with parametric amplitude and
    full double-scatter completeness.

    .. math::

        C_\ell^{\rm unres} = \int dz \int d\ln M \;
            \bigl[y_\ell^2 \; dn/d\ln M \; dV/dz/d\Omega\bigr]
            \times \bigl[1 - P_{\rm det}(M,z)\bigr]

    Both the tSZ integrand (via ratio^2) and the SNR entering P_det
    (via ratio) use the parametric amplitude.

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

    # --- Parametric SNR grid -> detection probability ---
    qbar_grid = build_snr_grid_parametric(
        m_grid, z_grid, A_SZ, alpha_SZ, params_values_dict,
        sigma_obj_file=sigma_obj_file,
        skyfr_file=skyfr_file,
        filter_name=filter_name,
        theta_min=theta_min,
        theta_max=theta_max,
    )

    P_det = _pdet_grid(
        qbar_grid, sigma_lnY=sigma_lnY, q_cat=q_cat,
        n_grid=n_grid, nsig=nsig,
    )

    # --- Unresolved mask: keep undetected halos ---
    mask_mz = 1.0 - P_det
    integrand = integrand * mask_mz[:, :, None]

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
    n_grid=4096,
    nsig=16.0,
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
    n_grid=4096,
    nsig=16.0,
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

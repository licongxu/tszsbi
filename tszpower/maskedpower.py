from .profiles import mpc_per_h_to_cm
import jax
import jax.numpy as jnp
from jax import lax
from . import classy_sz
from .utils import get_ell_range, simpson, get_ell_binwidth
from .tsz import get_integral_grid, get_integrand_number_counts, dVdzdOmega
from .profiles import y_ell_interpolate
from .massfuncs import get_hmf_at_z_and_m
from .utils import get_ell_range, simpson, get_ell_binwidth
import jax.scipy.special as jsp
from functools import partial


def compute_y0(M, z, params_values_dict=None):

    rparams = classy_sz.get_all_relevant_params(params_values_dict=params_values_dict)
    conv_fac = 299792.458  # km/s -> (km/s), used as c to scale H(z); ratio H/H0 is dimensionless
    h = rparams['H0'] / 100.0
    H = classy_sz.get_hubble_at_z(z, params_values_dict=params_values_dict) * conv_fac
    H0 = rparams['H0']
    B = rparams['B']
    c500 = rparams['c500']
    gamma = rparams['gammaGNFW']
    alpha = rparams['alphaGNFW']
    beta = rparams['betaGNFW']
    P0 = rparams['P0GNFW']

    # --- GNFW normalization at r500 ---
    m_delta_tilde = (M / B)  # to M_sun
    C = (
        1.65
        * (h / 0.7) ** 2
        * (H / H0) ** (8.0 / 3.0)
        * (m_delta_tilde / (0.7 * 3e14)) ** (2.0 / 3.0 + 0.12)
        * (0.7 / h) ** 1.5
    )  # eV cm^-3
    P500 = C * P0  # eV cm^-3

    # --- r500 and conversion to cm ---
    r500 = classy_sz.get_r_delta_of_m_delta_at_z(delta=500, m_delta=M, z=z, params_values_dict=params_values_dict)  # (assumed Mpc/h)
    r500_cm = mpc_per_h_to_cm(r500, h)

    # --- constants for y ---
    me_in_eV = 510_998.95     # eV
    sigmat_cm = 6.6524587321e-25  # cm^2
    prefactor = 2.0 * (sigmat_cm / me_in_eV) * P500 * r500_cm

    # --- GNFW shape integral ∫_0^∞ x^{-γ} (1 + x^α)^{(γ-β)/α} dx with x=r/r500 ---
    # def integrand(x):
    #     sx = c500 * x
    #     return (sx ** (-gamma)) * (1.0 + sx ** alpha) ** ((gamma - beta) / alpha)

    # integral, err = quad(integrand, 0.0, np.inf, epsabs=0.0, epsrel=1e-8, limit=500)
    integral = 0.470502095
    y0 = prefactor * integral

    return y0

def compute_theta500_arcmin(M, z, params_values_dict=None, delta = 500):
    dA_z = classy_sz.get_angular_distance_at_z(z, params_values_dict=params_values_dict) 
    r500 = classy_sz.get_r_delta_of_m_delta_at_z(delta=delta, m_delta=M, z=z, params_values_dict=params_values_dict) 
    theta_500_arcmin = (r500/dA_z) * (180./jnp.pi) * 60.  # convert rad to arcmin
    return theta_500_arcmin

def compute_sigma_y0(M, z, params_values_dict, 
                     sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy", 
                     skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
                     filter_name='immf6', theta_min=0.5, theta_max=32.0):
    """
    Vectorized version.
    Inputs:
      M, z : scalars or 1D arrays with the same shape
    Returns:
      sigma_at_theta500 : same shape as inputs
      theta_500_arcmin  : same shape as inputs
    """

    # Load sigma curves & sky fractions (once)
    sigma_obj = jnp.load(sigma_obj_file, allow_pickle=True).item()  # {filter_name -> {tile_index -> 1D array over theta_500}}
    skyfr = jnp.load(skyfr_file).ravel()                            # shape (num_tiles,)

    # Reconstruct theta grid and sky-avg curve
    data = sigma_obj[filter_name]               # {tile_index -> 1D array over theta_500}
    first = next(iter(data.values()))
    ntheta = len(first)
    theta_grid = jnp.exp(jnp.linspace(jnp.log(theta_min), jnp.log(theta_max), ntheta))  # positive

    num = jnp.zeros(ntheta, dtype=float)
    den = jnp.array(0.0, dtype=float)
    for tile, arr in data.items():
        w = skyfr[int(tile)]
        y = jnp.asarray(arr, dtype=float)
        num += w * y
        den += w
    sigma_skyavg = num / den  # shape (ntheta,)

    # ---- LOG–LOG interpolation with linear extrapolation ----
    eps = 1e-20  # tiny floor to avoid log(0)
    log_theta = jnp.log(theta_grid)                           # strictly increasing
    log_sigma = jnp.log(jnp.clip(sigma_skyavg, eps, None))    # finite

    def interp_loglog_extrap(xg, yg, xq):
        """
        xg, yg: 1D arrays (monotonic xg)
        xq: any shape; returns yg(xq) using piecewise-linear interpolation in (xg, yg),
            extrapolating linearly beyond the ends using the end segments.
        """
        xq = jnp.asarray(xq)
        # indices i with xg[i] <= xq < xg[i+1]; clamp to [0, len-2] for extrapolation
        i = jnp.searchsorted(xg, xq, side='right') - 1
        i = jnp.clip(i, 0, xg.size - 2)

        x0 = xg[i]
        x1 = xg[i + 1]
        y0 = yg[i]
        y1 = yg[i + 1]

        t = (xq - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    # Ensure array inputs; remember if inputs were scalar for output formatting
    M_arr = jnp.atleast_1d(M)
    z_arr = jnp.atleast_1d(z)
    if M_arr.shape != z_arr.shape:
        raise ValueError(f"M and z must have the same shape; got {M_arr.shape} vs {z_arr.shape}")

    # Vectorized theta_500(M,z)
    theta_500_arcmin = jax.vmap(lambda m, zz: compute_theta500_arcmin(m, zz, params_values_dict))(M_arr, z_arr)

    # Query in log-theta, interpolate/extrapolate in log–log, then exp back
    xi = jnp.log(theta_500_arcmin)
    log_sigma_q = interp_loglog_extrap(log_theta, log_sigma, xi)
    sigma_at_theta500 = jnp.exp(log_sigma_q).reshape(theta_500_arcmin.shape)

    # Scalar in → scalar out
    if jnp.ndim(M) == 0 and jnp.ndim(z) == 0:
        return sigma_at_theta500[0], theta_500_arcmin[0]
    else:
        return sigma_at_theta500, theta_500_arcmin



def compute_snr(M, z, params_values_dict,
                sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
                skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
                filter_name="immf6", theta_min=0.5, theta_max=32.0):
    """
    Compute the signal-to-noise ratio (SNR) for a cluster of mass M at redshift z.

    SNR = y0 / sigma_y0, where:
      - y0 is the central Compton-y amplitude from the GNFW profile.
      - sigma_y0 is the sky-averaged noise at theta_500 (with linear extrapolation).

    Returns
    -------
    snr : float
        The signal-to-noise ratio.
    y0 : float
        The Compton-y signal amplitude.
    sigma0 : float
        The noise level at theta_500.
    theta500_arcmin : float
        The cluster angular size in arcmin.
    """
    # signal
    y0 = compute_y0(M, z, params_values_dict=params_values_dict)

    # noise and cluster angular size
    sigma0, theta500_arcmin = compute_sigma_y0(
        M, z, params_values_dict=params_values_dict,
        sigma_obj_file=sigma_obj_file,
        skyfr_file=skyfr_file,
        filter_name=filter_name,
        theta_min=theta_min,
        theta_max=theta_max,
    )

    snr = y0 / sigma0
    return snr


@jax.jit
def theta_q_minus_snr(snr_grid, q_obs, at=0.0):
    """
    Compute Θ(q_obs - snr) elementwise.

    Parameters
    ----------
    snr_grid : jnp.ndarray
        SNR evaluated on the (z, M) grid. Any shape is fine (e.g. (n_z, n_m)).
    q_obs : float or array broadcastable to snr_grid
        Detection threshold q_obs.
    at : float, default 0.0
        Value to return when q_obs - snr == 0 (convention choice).

    Returns
    -------
    mask : jnp.ndarray
        Same shape as snr_grid, with values in {0, at, 1}.
    """
    q = jnp.asarray(q_obs, dtype=snr_grid.dtype)
    return jnp.heaviside(q - snr_grid, at)


def build_snr_grid(
    m_grid, z_grid, params_values_dict,
    *,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
):
    """
    Return snr_grid with shape (n_z, n_m), matching (z_grid, m_grid).
    """
    m_grid = jnp.asarray(m_grid)
    z_grid = jnp.asarray(z_grid)

    def snr_at_z(zz):
        # vector over m_grid for fixed z
        return jax.vmap(lambda mm: compute_snr(
            mm, zz, params_values_dict,
            sigma_obj_file=sigma_obj_file,
            skyfr_file=skyfr_file,
            filter_name=filter_name,
            theta_min=theta_min,
            theta_max=theta_max,
        ))(m_grid)

    # vector over z_grid
    return jax.vmap(snr_at_z)(z_grid)  # (n_z, n_m)


@jax.jit
def _integrate_mz(integrand, z_grid, logm_grid):
    """
    JIT-compiled integration over m and z for each ell slice.
    integrand: (n_z, n_m, n_ell)
    z_grid   : (n_z,)
    logm_grid: (n_m,)
    """
    dx_m = logm_grid[1] - logm_grid[0]
    dx_z = z_grid[1] - z_grid[0]

    ell = get_ell_range()          # (n_ell,)
    n_ell = ell.shape[0]

    def scan_body(_, i):
        integrand_i = integrand[:, :, i]                           # (n_z, n_m)
        partial_m = simpson(integrand_i, x=logm_grid, dx=dx_m, axis=1)  # (n_z,)
        result    = simpson(partial_m,   x=z_grid,    dx=dx_z,   axis=0)  # scalar
        return None, result

    _, C_yy = lax.scan(scan_body, None, jnp.arange(n_ell))
    return C_yy  # (n_ell,)


@partial(
    jax.jit,
    static_argnames=(
        "sigma_obj_file",
        "skyfr_file",
        "filter_name",
        "theta_min",
        "theta_max",
    ),
)
def compute_integral_snr_simple_uRC(
    params_values_dict=None,
    *,
    q_obs=None,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
):
    """
    If q_obs is provided, applies the step-function selection Θ(q_obs - snr(M,z))
    to the integrand before integrating over m and z.
    """
    # --- reproduce the exact grids used in get_integral_grid
    allparams = classy_sz.get_all_relevant_params(params_values_dict=params_values_dict)
    z_min, z_max = allparams['z_min'], allparams['z_max']
    z_grid = jnp.geomspace(z_min, z_max, 100)

    M_min, M_max = allparams['M_min'], allparams['M_max']
    m_grid = jnp.geomspace(M_min, M_max, 100)
    logm_grid = jnp.log(m_grid)

    # --- base integrand (n_z, n_m, n_ell)
    integrand = get_integral_grid(params_values_dict=params_values_dict)

    # --- optional selection: Θ(q_obs - snr(M,z))
    if q_obs is not None:
        snr_grid = build_snr_grid(
            m_grid, z_grid, params_values_dict,
            sigma_obj_file=sigma_obj_file,
            skyfr_file=skyfr_file,
            filter_name=filter_name,
            theta_min=theta_min,
            theta_max=theta_max,
        )  # (n_z, n_m)

        mask_mz = theta_q_minus_snr(snr_grid, q_obs)  # (n_z, n_m)
        integrand = integrand * mask_mz[:, :, None]   # broadcast over ell
    # --- integrate over m and z for each ell (JIT)
    C_yy = _integrate_mz(integrand, z_grid, logm_grid)
    return C_yy


def compute_integral_snr_simple_RC(
    params_values_dict=None,
    *,
    q_obs=None,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
):
    """
    If q_obs is provided, applies the step-function selection Θ(q_obs - snr(M,z))
    to the integrand before integrating over m and z.
    """
    # --- reproduce the exact grids used in get_integral_grid
    allparams = classy_sz.get_all_relevant_params(params_values_dict=params_values_dict)
    z_min, z_max = allparams['z_min'], allparams['z_max']
    z_grid = jnp.geomspace(z_min, z_max, 100)

    M_min, M_max = allparams['M_min'], allparams['M_max']
    m_grid = jnp.geomspace(M_min, M_max, 100)
    logm_grid = jnp.log(m_grid)

    # --- base integrand (n_z, n_m, n_ell)
    integrand = get_integral_grid(params_values_dict=params_values_dict)

    # --- optional selection: Θ(q_obs - snr(M,z))
    if q_obs is not None:
        snr_grid = build_snr_grid(
            m_grid, z_grid, params_values_dict,
            sigma_obj_file=sigma_obj_file,
            skyfr_file=skyfr_file,
            filter_name=filter_name,
            theta_min=theta_min,
            theta_max=theta_max,
        )  # (n_z, n_m)

        mask_mz = 1 - theta_q_minus_snr(snr_grid, q_obs)  # (n_z, n_m)
        integrand = integrand * mask_mz[:, :, None]   # broadcast over ell

    # --- integrate over m and z for each ell (JIT)
    C_yy = _integrate_mz(integrand, z_grid, logm_grid)
    return C_yy


@jax.jit
def compute_number_resolved_clusters(
    params_values_dict=None,
    *,
    q_obs=None,
    f_sky=1.0,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
):
    """
    Compute the number of resolved clusters (SNR > q_obs) scaled by f_sky.

    Returns N_resolved = 4π * f_sky * ∫ dz ∫ dlnM [ (dV/dz/dΩ)(z) * d n / d ln M (M,z) ] * H(SNR(M,z) - q_obs).

    If q_obs is None, returns the total number of clusters (no SNR cut).
    """
    # --- define z and mass grids (same as tSZ integrals) ---
    allparams = classy_sz.get_all_relevant_params(params_values_dict=params_values_dict)
    z_min, z_max = allparams['z_min'], allparams['z_max']
    z_grid = jnp.geomspace(z_min, z_max, 100)

    M_min, M_max = allparams['M_min'], allparams['M_max']
    m_grid = jnp.geomspace(M_min, M_max, 100)
    logm_grid = jnp.log(m_grid)

    dx_m = logm_grid[1] - logm_grid[0]
    dx_z = z_grid[1] - z_grid[0]

    # --- base integrand: 4π * (dV/dz/dΩ) * (dn/dlnM) ---
    integrand_N = get_integrand_number_counts(params_values_dict=params_values_dict)  # (n_z, n_m, 1)
    integrand_N = integrand_N[:, :, 0]  # drop dummy axis

    # --- optional SNR > q_obs selection ---
    if q_obs is not None:
        snr_grid = build_snr_grid(
            m_grid,
            z_grid,
            params_values_dict,
            sigma_obj_file=sigma_obj_file,
            skyfr_file=skyfr_file,
            filter_name=filter_name,
            theta_min=theta_min,
            theta_max=theta_max,
        )  # (n_z, n_m)

        # theta_q_minus_snr = H(q_obs - SNR)
        # 1 - that = H(SNR - q_obs)
        mask_resolved = 1.0 - theta_q_minus_snr(snr_grid, q_obs)
        integrand_N = integrand_N * mask_resolved

    # --- integrate over ln M then z (Simpson integration) ---
    partial_m = simpson(integrand_N, x=logm_grid, dx=dx_m, axis=1)  # (n_z,)
    N_res_fullsky = simpson(partial_m, x=z_grid, dx=dx_z, axis=0)   # scalar

    # --- scale by sky fraction ---
    N_res = f_sky * N_res_fullsky

    return N_res

@jax.jit
def compute_cluster_counts_in_z_snr_bins(
    params_values_dict=None,
    *,
    z_bin_edges=None,      # 1D array of size (n_z_bins + 1)
    snr_bin_edges=None,    # 1D array of size (n_snr_bins + 1), in SNR (not ln SNR)
    f_sky=1.0,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
):
    """
    Compute the number of clusters in each (z, SNR) bin.

    Returns
    -------
    N_bins : jnp.ndarray, shape (n_z_bins, n_snr_bins)
        N_bins[i, j] = number of clusters in:
            z ∈ [z_bin_edges[i],   z_bin_edges[i+1]),
            SNR ∈ [snr_bin_edges[j], snr_bin_edges[j+1]),
        scaled by f_sky (4π already included in get_integrand_number_counts).
    """

    if z_bin_edges is None or snr_bin_edges is None:
        raise ValueError("z_bin_edges and snr_bin_edges must be provided.")

    z_bin_edges = jnp.asarray(z_bin_edges)
    snr_bin_edges = jnp.asarray(snr_bin_edges)

    # --- reproduce grids used in your tSZ & SNR integrals ---
    allparams = classy_sz.get_all_relevant_params(params_values_dict=params_values_dict)
    z_min, z_max = allparams['z_min'], allparams['z_max']
    z_grid = jnp.geomspace(z_min, z_max, 100)              # (n_z,)

    M_min, M_max = allparams['M_min'], allparams['M_max']
    m_grid = jnp.geomspace(M_min, M_max, 100)              # (n_m,)
    logm_grid = jnp.log(m_grid)

    dx_m = logm_grid[1] - logm_grid[0]
    dx_z = z_grid[1] - z_grid[0]

    # --- base integrand: 4π * dV/dz/dΩ * dn/dlnM ---
    # get_integrand_number_counts returns shape (n_z, n_m, 1)
    integrand_full = get_integrand_number_counts(params_values_dict=params_values_dict)
    integrand_N = integrand_full[:, :, 0]   # (n_z, n_m), full-sky

    # --- SNR grid on same (z, M) grid ---
    snr_grid = build_snr_grid(
        m_grid,
        z_grid,
        params_values_dict,
        sigma_obj_file=sigma_obj_file,
        skyfr_file=skyfr_file,
        filter_name=filter_name,
        theta_min=theta_min,
        theta_max=theta_max,
    )  # shape (n_z, n_m)

    # --- build all bin combinations (z_i, snr_j) ---
    z_min_edges = z_bin_edges[:-1]
    z_max_edges = z_bin_edges[1:]
    q_min_edges = snr_bin_edges[:-1]
    q_max_edges = snr_bin_edges[1:]

    # meshgrid to get all (z_bin, snr_bin) pairs
    Zmin, Qmin = jnp.meshgrid(z_min_edges, q_min_edges, indexing="ij")
    Zmax, Qmax = jnp.meshgrid(z_max_edges, q_max_edges, indexing="ij")

    Zmin_flat = Zmin.ravel()
    Zmax_flat = Zmax.ravel()
    Qmin_flat = Qmin.ravel()
    Qmax_flat = Qmax.ravel()

    def count_one_bin(z_lo, z_hi, q_lo, q_hi):
        """
        Integrate over z and lnM with masks:
          z_lo <= z < z_hi
          q_lo <= SNR < q_hi
        """
        # mask in z
        mask_z = (z_grid >= z_lo) & (z_grid < z_hi)        # (n_z,)
        mask_z = mask_z[:, None]                           # (n_z, 1)

        # mask in SNR
        mask_snr = (snr_grid >= q_lo) & (snr_grid < q_hi)  # (n_z, n_m)

        mask = mask_z & mask_snr                           # (n_z, n_m)
        mask = mask.astype(integrand_N.dtype)

        integrand_bin = integrand_N * mask                 # (n_z, n_m)

        # integrate over lnM then z
        partial_m = simpson(integrand_bin, x=logm_grid, dx=dx_m, axis=1)  # (n_z,)
        N_fullsky_bin = simpson(partial_m, x=z_grid, dx=dx_z, axis=0)     # scalar

        return f_sky * N_fullsky_bin

    # vmap over all flattened bin pairs
    count_vmap = jax.vmap(count_one_bin)
    N_flat = count_vmap(Zmin_flat, Zmax_flat, Qmin_flat, Qmax_flat)

    # reshape back to (n_z_bins, n_snr_bins)
    n_z_bins = z_bin_edges.size - 1
    n_snr_bins = snr_bin_edges.size - 1
    N_bins = N_flat.reshape((n_z_bins, n_snr_bins))

    return N_bins


def compute_tsz_covariance_masked_snr(
    params_values_dict=None,
    *,
    q_obs=None,
    noise_ell=None,
    f_sky=1.0,
    # SNR/noise curve inputs (same defaults as your masked-power functions)
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
):
    """
    Covariance matrix for an SNR-masked (unresolved) tSZ map, including 1-halo trispectrum.

    Mask convention:
        W(M,z;q) = Θ(q - SNR(M,z))   -> keeps UNRESOLVED objects (SNR < q)

    Returns
    -------
    ell_arr : (n_ell,) jnp.ndarray
    M      : (n_ell, n_ell) jnp.ndarray   full covariance (Gaussian + trispectrum)
    M_G    : (n_ell, n_ell) jnp.ndarray   Gaussian-only covariance
    """
    import jax
    import jax.numpy as jnp
    from jax import lax

    from .utils import get_ell_range, simpson, get_ell_binwidth
    from .tsz import dVdzdOmega
    from .profiles import y_ell_interpolate
    from .massfuncs import get_hmf_at_z_and_m
    from .tsz import get_integral_grid  # unmasked integrand grid

    # ------------------------------------------------------------------
    # 0) Precompute sigma_skyavg(theta) OUTSIDE jit (strings live here)
    # ------------------------------------------------------------------
    sigma_obj = jnp.load(sigma_obj_file, allow_pickle=True).item()
    skyfr = jnp.load(skyfr_file).ravel()

    data = sigma_obj[filter_name]
    first = next(iter(data.values()))
    ntheta = len(first)

    theta_grid = jnp.exp(jnp.linspace(jnp.log(theta_min), jnp.log(theta_max), ntheta))
    num = jnp.zeros(ntheta, dtype=float)
    den = jnp.array(0.0, dtype=float)
    for tile, arr in data.items():
        w = skyfr[int(tile)]
        y = jnp.asarray(arr, dtype=float)
        num += w * y
        den += w
    sigma_skyavg = num / den  # (ntheta,)

    log_theta_grid = jnp.log(theta_grid)
    eps = 1e-20
    log_sigma_skyavg = jnp.log(jnp.clip(sigma_skyavg, eps, None))

    # ------------------------------------------------------------------
    # 1) JAX-safe log-log interp/extrap + sigma(theta500)
    # ------------------------------------------------------------------
    @jax.jit
    def _interp_lin_extrap(xg, yg, xq):
        # piecewise linear interpolation with linear extrapolation
        i = jnp.searchsorted(xg, xq, side="right") - 1
        i = jnp.clip(i, 0, xg.size - 2)
        x0 = xg[i]
        x1 = xg[i + 1]
        y0 = yg[i]
        y1 = yg[i + 1]
        t = (xq - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    @jax.jit
    def sigma_y0_from_theta500(theta500_arcmin):
        # log-log interpolation/extrapolation
        xi = jnp.log(theta500_arcmin)
        log_sig = _interp_lin_extrap(log_theta_grid, log_sigma_skyavg, xi)
        return jnp.exp(log_sig)

    @jax.jit
    def theta_q_minus_snr(snr_grid, q, at=0.0):
        q = jnp.asarray(q, dtype=snr_grid.dtype)
        return jnp.heaviside(q - snr_grid, at)

    # ------------------------------------------------------------------
    # 2) Grids (match your integrals)
    # ------------------------------------------------------------------
    allparams = classy_sz.get_all_relevant_params(params_values_dict=params_values_dict)
    z_grid = jnp.geomspace(allparams["z_min"], allparams["z_max"], 100)  # (n_z,)
    m_grid = jnp.geomspace(allparams["M_min"], allparams["M_max"], 100)  # (n_m,)
    logm_grid = jnp.log(m_grid)

    ell_arr = get_ell_range()
    n_ell = ell_arr.size

    # ------------------------------------------------------------------
    # 3) Build SNR grid on (z,m) using the precomputed sigma curve
    # ------------------------------------------------------------------
    def snr_at_one(m, zz):
        y0 = compute_y0(m, zz, params_values_dict=params_values_dict)
        th = compute_theta500_arcmin(m, zz, params_values_dict=params_values_dict)
        sig = sigma_y0_from_theta500(th)
        return y0 / sig

    def snr_at_z(zz):
        return jax.vmap(lambda mm: snr_at_one(mm, zz))(m_grid)  # (n_m,)

    snr_grid = jax.vmap(snr_at_z)(z_grid)  # (n_z, n_m)

    if q_obs is None:
        mask_mz = jnp.ones_like(snr_grid, dtype=float)
    else:
        mask_mz = theta_q_minus_snr(snr_grid, q_obs)  # Θ(q - SNR)

    # ------------------------------------------------------------------
    # 4) Masked power spectrum C_ell(q): apply mask to your unmasked integrand grid
    # ------------------------------------------------------------------
    integrand_C = get_integral_grid(params_values_dict=params_values_dict)  # (n_z,n_m,n_ell)
    integrand_C = integrand_C * mask_mz[:, :, None]

    @jax.jit
    def _integrate_mz(integrand, z_grid, logm_grid):
        dx_m = logm_grid[1] - logm_grid[0]
        dx_z = z_grid[1] - z_grid[0]

        def scan_body(_, i):
            integ_i = integrand[:, :, i]  # (n_z,n_m)
            partial_m = simpson(integ_i, x=logm_grid, dx=dx_m, axis=1)  # (n_z,)
            res = simpson(partial_m, x=z_grid, dx=dx_z, axis=0)         # scalar
            return None, res

        _, C = lax.scan(scan_body, None, jnp.arange(integrand.shape[-1]))
        return C

    C_yy_mask = _integrate_mz(integrand_C, z_grid, logm_grid)  # (n_ell,)

    # ------------------------------------------------------------------
    # 5) Masked trispectrum T_{ell,ell'}(q) (1-halo): also apply same mask_mz
    # ------------------------------------------------------------------
    def get_yellm_for_z(zp):
        _, y_ellm = y_ell_interpolate(zp, m_grid, params_values_dict=params_values_dict)
        return y_ellm  # (n_m,n_ell)

    y_ell_mz = jax.vmap(get_yellm_for_z)(z_grid)  # (n_z,n_m,n_ell)
    y_sq = y_ell_mz**2

    def get_hmf_for_z(zp):
        return get_hmf_at_z_and_m(z=zp, m=m_grid, params_values_dict=params_values_dict)

    dndlnm = jax.vmap(get_hmf_for_z)(z_grid)  # (n_z,n_m)
    comov = dVdzdOmega(z_grid, params_values_dict=params_values_dict)  # (n_z,)

    integrand_T = (
        y_sq[:, :, :, None] * y_sq[:, :, None, :] *
        dndlnm[:, :, None, None] *
        comov[:, None, None, None] *
        mask_mz[:, :, None, None]
    )  # (n_z,n_m,n_ell,n_ell)

    partial_m = simpson(integrand_T, x=logm_grid, axis=1)  # (n_z,n_ell,n_ell)
    T_mask = simpson(partial_m, x=z_grid, axis=0)          # (n_ell,n_ell)

    # ------------------------------------------------------------------
    # 6) Assemble covariance (match your tsz.compute_tsz_covariance convention)
    # ------------------------------------------------------------------
    delta_ell = get_ell_binwidth()
    if noise_ell is None:
        noise_ell = jnp.zeros_like(C_yy_mask)

    diag_term = (4.0 * jnp.pi) * (C_yy_mask + noise_ell) ** 2 / (ell_arr + 0.5)

    M_G = jnp.diag(diag_term) / (4.0 * jnp.pi * f_sky * delta_ell)
    M = M_G + T_mask / (4.0 * jnp.pi * f_sky)

    return ell_arr, M, M_G


def compute_scaled_tsz_covariance_masked_snr(
    params_values_dict=None,
    *,
    q_obs=None,
    noise_ell=None,
    f_sky=1.0,
    # SNR/noise curve inputs (same defaults as your masked-power functions)
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
):
    """
    Scaled covariance for D_ell = ell(ell+1) C_ell / (2π), on an SNR-masked (unresolved) tSZ map.

    Returns
    -------
    ell_arr : (n_ell,)
    M_D     : (n_ell, n_ell)  covariance of D_ell (Gaussian + trispectrum)
    M_DG    : (n_ell, n_ell)  Gaussian-only covariance of D_ell
    """
    # First compute covariance of C_ell on the masked map
    ell_arr, M_C, M_CG = compute_tsz_covariance_masked_snr(
        params_values_dict=params_values_dict,
        q_obs=q_obs,
        noise_ell=noise_ell,
        f_sky=f_sky,
        sigma_obj_file=sigma_obj_file,
        skyfr_file=skyfr_file,
        filter_name=filter_name,
        theta_min=theta_min,
        theta_max=theta_max,
    )

    # Scale matrix: D_ell = s_ell * C_ell, with s_ell = ell(ell+1)/(2π)
    s = ell_arr * (ell_arr + 1.0) / (2.0 * jnp.pi)  # (n_ell,)
    S2D = jnp.outer(s, s)                            # (n_ell, n_ell)

    # Cov(D) = S * Cov(C) * S  -> elementwise since S is diagonal: Cov_ij(D)=s_i s_j Cov_ij(C)
    M_D  = S2D * M_C
    M_DG = S2D * M_CG

    # Optional: match your "scaled trispectrum" convention multiplying by 1e24
    # (Only do this if your D_ell is also in that unit system.)
    M_D  = M_D * 1e24
    M_DG = M_DG * 1e24

    return ell_arr, M_D, M_DG


def compute_trispectrum_masked_snr(
    params_values_dict=None,
    *,
    q_obs=None,
    # SNR/noise curve inputs (same defaults as your masked-power functions)
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
):
    """
    Trispectrum-only (1-halo) for an SNR-masked (unresolved) tSZ map.

    Mask convention:
        W(M,z;q) = Θ(q - SNR(M,z))   -> keeps UNRESOLVED objects (SNR < q)

    Returns
    -------
    ell_arr : (n_ell,) jnp.ndarray
    T_mask  : (n_ell, n_ell) jnp.ndarray   trispectrum (no Gaussian term)
    """
    import jax
    import jax.numpy as jnp

    # --- 0) Precompute sigma_skyavg(theta) OUTSIDE jit ---
    sigma_obj = jnp.load(sigma_obj_file, allow_pickle=True).item()
    skyfr = jnp.load(skyfr_file).ravel()

    data = sigma_obj[filter_name]
    first = next(iter(data.values()))
    ntheta = len(first)

    theta_grid = jnp.exp(jnp.linspace(jnp.log(theta_min), jnp.log(theta_max), ntheta))
    num = jnp.zeros(ntheta, dtype=float)
    den = jnp.array(0.0, dtype=float)
    for tile, arr in data.items():
        w = skyfr[int(tile)]
        y = jnp.asarray(arr, dtype=float)
        num += w * y
        den += w
    sigma_skyavg = num / den

    log_theta_grid = jnp.log(theta_grid)
    eps = 1e-20
    log_sigma_skyavg = jnp.log(jnp.clip(sigma_skyavg, eps, None))

    # --- 1) JAX-safe log-log interp/extrap (same as in your covariance func) ---
    @jax.jit
    def _interp_lin_extrap(xg, yg, xq):
        i = jnp.searchsorted(xg, xq, side="right") - 1
        i = jnp.clip(i, 0, xg.size - 2)
        x0 = xg[i]
        x1 = xg[i + 1]
        y0 = yg[i]
        y1 = yg[i + 1]
        t = (xq - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    @jax.jit
    def sigma_y0_from_theta500(theta500_arcmin):
        xi = jnp.log(theta500_arcmin)
        log_sig = _interp_lin_extrap(log_theta_grid, log_sigma_skyavg, xi)
        return jnp.exp(log_sig)

    @jax.jit
    def _theta_q_minus_snr(snr_grid, q, at=0.0):
        q = jnp.asarray(q, dtype=snr_grid.dtype)
        return jnp.heaviside(q - snr_grid, at)

    # --- 2) Grids (match your integrals) ---
    allparams = classy_sz.get_all_relevant_params(params_values_dict=params_values_dict)
    z_grid = jnp.geomspace(allparams["z_min"], allparams["z_max"], 100)
    m_grid = jnp.geomspace(allparams["M_min"], allparams["M_max"], 100)
    logm_grid = jnp.log(m_grid)

    ell_arr = get_ell_range()

    # --- 3) Build SNR grid and mask_mz ---
    def snr_at_one(m, zz):
        y0 = compute_y0(m, zz, params_values_dict=params_values_dict)
        th = compute_theta500_arcmin(m, zz, params_values_dict=params_values_dict)
        sig = sigma_y0_from_theta500(th)
        return y0 / sig

    def snr_at_z(zz):
        return jax.vmap(lambda mm: snr_at_one(mm, zz))(m_grid)

    snr_grid = jax.vmap(snr_at_z)(z_grid)  # (n_z, n_m)

    if q_obs is None:
        mask_mz = jnp.ones_like(snr_grid, dtype=float)
    else:
        mask_mz = _theta_q_minus_snr(snr_grid, q_obs)

    # --- 4) Build trispectrum integrand and integrate ---
    def get_yellm_for_z(zp):
        _, y_ellm = y_ell_interpolate(zp, m_grid, params_values_dict=params_values_dict)
        return y_ellm  # (n_m, n_ell)

    y_ell_mz = jax.vmap(get_yellm_for_z)(z_grid)  # (n_z, n_m, n_ell)
    y_sq = y_ell_mz**2

    def get_hmf_for_z(zp):
        return get_hmf_at_z_and_m(z=zp, m=m_grid, params_values_dict=params_values_dict)

    dndlnm = jax.vmap(get_hmf_for_z)(z_grid)  # (n_z, n_m)
    comov = dVdzdOmega(z_grid, params_values_dict=params_values_dict)  # (n_z,)

    integrand_T = (
        y_sq[:, :, :, None] * y_sq[:, :, None, :] *
        dndlnm[:, :, None, None] *
        comov[:, None, None, None] *
        mask_mz[:, :, None, None]
    )  # (n_z, n_m, n_ell, n_ell)

    partial_m = simpson(integrand_T, x=logm_grid, axis=1)  # (n_z, n_ell, n_ell)
    T_mask = simpson(partial_m, x=z_grid, axis=0)          # (n_ell, n_ell)

    return ell_arr, T_mask


def compute_scaled_trispectrum_masked_snr(
    params_values_dict=None,
    *,
    q_obs=None,
    # SNR/noise curve inputs
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
):
    """
    Scaled trispectrum-only for D_ell = ell(ell+1)C_ell/(2π), on an SNR-masked map.

    Returns
    -------
    ell_arr : (n_ell,)
    scaled_T : (n_ell, n_ell)  where
        scaled_T_{ll'} = T_{ll'} * [l(l+1) l'(l'+1)] / (2π)^2
    and then multiplied by 1e24 (to match your convention).
    """
    ell_arr, T_mask = compute_trispectrum_masked_snr(
        params_values_dict=params_values_dict,
        q_obs=q_obs,
        sigma_obj_file=sigma_obj_file,
        skyfr_file=skyfr_file,
        filter_name=filter_name,
        theta_min=theta_min,
        theta_max=theta_max,
    )

    ell_factor = ell_arr * (ell_arr + 1.0)  # (n_ell,)
    ell2D_factor = jnp.outer(ell_factor, ell_factor)  # (n_ell, n_ell)

    prefactor = 1.0 / ((2.0 * jnp.pi) ** 2)
    scaled_T = T_mask * ell2D_factor * prefactor

    return ell_arr, scaled_T * 1e24

def compute_Dell_snr_simple_uRC(
    params_values_dict=None,
    *,
    q_obs=None,
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
):
    """
    Compute D_ell for the SNR-masked (unresolved) tSZ map.

    Uses the same selection as compute_integral_snr_simple_uRC:
        W(M,z;q) = Θ(q - SNR(M,z))  (keeps SNR < q)

    Returns
    -------
    D_ell_yy : (n_ell,) jnp.ndarray
        D_ell = ell(ell+1) C_ell / (2π) * 1e12
    """
    ell = get_ell_range()

    # C_ell for the unresolved (masked) map
    C_ell_yy = compute_integral_snr_simple_uRC(
        params_values_dict=params_values_dict,
        q_obs=q_obs,
        sigma_obj_file=sigma_obj_file,
        skyfr_file=skyfr_file,
        filter_name=filter_name,
        theta_min=theta_min,
        theta_max=theta_max,
    )

    # Convert to D_ell with your convention
    D_ell_yy = ell * (ell + 1.0) * C_ell_yy / (2.0 * jnp.pi) * 1e12
    return D_ell_yy



@partial(jax.jit, static_argnames=("n_grid", "nsig"))
def completeness_convolution_jax(
    q_bar,          # \bar{q_m}
    sigma_lnY,      # scatter in ln space
    q_cat,          # catalog SNR threshold
    *,
    n_grid=1024,    # number of t points for integration
    nsig=16.0,       # integrate ±nsig*σ in ln q
):
    """
    Compute ∫_0^∞ dq_m f(q_m) g(q_cat - q_m) with
      f(q_m) = (1/q_m) exp( -[(ln(q_m/q_bar))^2 / (2 σ_lnY^2)])
      g(q_cat - q_m) = 1 - erf( (q_cat - q_m)/sqrt(2) ).
    Uses change of var t = ln q_m so dq_m/q_m = dt, then 
    ∫_{-∞}^{∞} dt e^{-(t-μ)^2/(2σ^2)} [1 - erf((q_cat - e^t)/√2)].

    Returns a scalar JAX array.
    """
    # safety
    q_bar      = jnp.maximum(q_bar,      1e-30)
    sigma_lnY  = jnp.maximum(sigma_lnY,  1e-12)

    # integration bounds in ln-space
    mu  = jnp.log(q_bar)
    t0  = mu - nsig * sigma_lnY
    t1  = mu + nsig * sigma_lnY

    # grid in ln-space
    t = jnp.linspace(t0, t1, n_grid)
    q_m = jnp.exp(t)

    # Gaussian in ln q
    expo = -0.5 * ((t - mu) / sigma_lnY) ** 2

    # selection factor
    arg  = (q_cat - q_m) / jnp.sqrt(2.0)
    g = 1.0 - jsp.erf(arg)

    # integrand in dt
    integrand = jnp.exp(expo) * g

    # trapezoidal rule (JAX API)
    return jnp.trapezoid(integrand, x=t)


# assumes you already defined:
# - build_snr_grid(m_grid, z_grid, params_values_dict, ...)
# - completeness_convolution_jax(q_bar, sigma_lnY, q_cat, n_grid=..., nsig=...)

@partial(jax.jit, static_argnames=("n_grid", "nsig"))
def compute_number_detected_clusters_completeness(
    params_values_dict=None,
    *,
    q_cat,                 # catalog SNR threshold (same role as q_obs)
    sigma_lnY,             # intrinsic ln-scatter (user-specified constant)
    f_sky=1.0,
    # SNR/noise-curve inputs (same defaults as your previous code)
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
    # completeness integral controls (must be static for stable JIT shapes)
    n_grid=4096,
    nsig=16.0,
):
    """
    Number of *detected* clusters using completeness:
        P_det(M,z) = ∫_0^∞ dq_m f(q_m | q_bar(M,z), sigma_lnY) g(q_cat - q_m)

    Then
        N_det = f_sky * ∫ dz ∫ dlnM  [dN/(dz dlnM)] * P_det(M,z)

    where dN/(dz dlnM) is provided by get_integrand_number_counts (full-sky, includes 4π).
    """

    # --- grids (match your other integrals) ---
    allparams = classy_sz.get_all_relevant_params(params_values_dict=params_values_dict)
    z_grid = jnp.geomspace(allparams["z_min"], allparams["z_max"], 100)  # (n_z,)
    m_grid = jnp.geomspace(allparams["M_min"], allparams["M_max"], 100)  # (n_m,)
    logm_grid = jnp.log(m_grid)

    dx_m = logm_grid[1] - logm_grid[0]
    dx_z = z_grid[1] - z_grid[0]

    # --- base number-count integrand: full-sky dN/(dz dlnM); shape (n_z, n_m) ---
    integrand_full = get_integrand_number_counts(params_values_dict=params_values_dict)
    dN_dzdlnM = integrand_full[:, :, 0]

    # --- mean SNR field q_bar(M,z): shape (n_z, n_m) ---
    qbar_grid = build_snr_grid(
        m_grid, z_grid, params_values_dict,
        sigma_obj_file=sigma_obj_file,
        skyfr_file=skyfr_file,
        filter_name=filter_name,
        theta_min=theta_min,
        theta_max=theta_max,
    )

    # --- completeness P_det(M,z) from convolution, vectorized over the grid ---
    qbar_flat = qbar_grid.reshape(-1)

    detprob_flat = jax.vmap(
        lambda qb: completeness_convolution_jax(
            qb, sigma_lnY, q_cat, n_grid=n_grid, nsig=nsig
        )
    )(qbar_flat)

    P_det = detprob_flat.reshape(qbar_grid.shape)  # (n_z, n_m)

    # --- integrate counts weighted by completeness ---
    integrand = dN_dzdlnM * P_det/(2*jnp.sqrt(2*jnp.pi)*sigma_lnY)  # (n_z, n_m)
    partial_m = simpson(integrand, x=logm_grid, dx=dx_m, axis=1)  # (n_z,)
    N_fullsky = simpson(partial_m, x=z_grid, dx=dx_z, axis=0)     # scalar

    return f_sky * N_fullsky


@partial(jax.jit, static_argnames=("n_grid", "nsig", "n_q_bins", "n_z_bins"))
def compute_cluster_counts_in_z_q_bins_completeness(
    params_values_dict=None,
    *,
    sigma_lnY,
    # requested defaults
    q_min=5.0,
    q_max=200.0,
    n_q_bins=20,
    z_min_bin=0.005,
    z_max_bin=3.0,
    n_z_bins=20,
    f_sky=1.0,
    # SNR/noise curve inputs
    sigma_obj_file="/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy",
    filter_name="immf6",
    theta_min=0.5,
    theta_max=32.0,
    # completeness integral controls
    n_grid=4096,
    nsig=16.0,
):
    """
    Returns:
      N_bins: (n_z_bins, n_q_bins)
      z_edges: (n_z_bins+1,)
      q_edges: (n_q_bins+1,)

    Here q is the *catalog SNR* threshold variable q_cat.
    Bin probability uses the correct ordering since P_det(q_cat) decreases with q_cat:
        P(q in [q_lo,q_hi)) = P_det(q_lo) - P_det(q_hi)
    """

    # ---- grids used for the M,z integration ----
    allparams = classy_sz.get_all_relevant_params(params_values_dict=params_values_dict)
    z_grid = jnp.geomspace(allparams["z_min"], allparams["z_max"], 100)
    m_grid = jnp.geomspace(allparams["M_min"], allparams["M_max"], 100)
    logm_grid = jnp.log(m_grid)

    dx_m = logm_grid[1] - logm_grid[0]
    dx_z = z_grid[1] - z_grid[0]

    # ---- base dN/(dz dlnM), full-sky (includes 4π) ----
    integrand_full = get_integrand_number_counts(params_values_dict=params_values_dict)
    dN_dzdlnM = integrand_full[:, :, 0]  # (n_z, n_m)

    # ---- qbar(M,z) grid ----
    qbar_grid = build_snr_grid(
        m_grid, z_grid, params_values_dict,
        sigma_obj_file=sigma_obj_file,
        skyfr_file=skyfr_file,
        filter_name=filter_name,
        theta_min=theta_min,
        theta_max=theta_max,
    )
    qbar_flat = qbar_grid.reshape(-1)

    # ---- bin edges (clip z-range to overlap the integration grid) ----
    z_lo_eff = jnp.maximum(z_min_bin, z_grid.min())
    z_hi_eff = jnp.minimum(z_max_bin, z_grid.max())
    z_edges = jnp.linspace(z_lo_eff, z_hi_eff, n_z_bins + 1)

    q_edges = jnp.linspace(q_min, q_max, n_q_bins + 1)

    # ---- completeness evaluated at all q_edges ----
    def Pdet_at_qedge(qc):
        return jax.vmap(
            lambda qb: completeness_convolution_jax(qb, sigma_lnY, qc, n_grid=n_grid, nsig=nsig)
        )(qbar_flat)

    Pdet_edges_flat = jax.vmap(Pdet_at_qedge)(q_edges)  # (n_q+1, n_z*n_m)
    Pdet_edges = Pdet_edges_flat.reshape((n_q_bins + 1,) + qbar_grid.shape)  # (n_q+1, n_z, n_m)

    # normalize exactly like your working N_det:
    # (your g ranges up to 2, hence the extra factor 2)
    norm = 2.0 * jnp.sqrt(2.0 * jnp.pi) * sigma_lnY
    Pdet_edges = Pdet_edges / norm

    # ---- probability mass in each q-bin [q_j, q_{j+1}) ----
    # IMPORTANT: Pdet decreases with q_cat, so use Pdet(q_lo) - Pdet(q_hi)
    Pbin = Pdet_edges[:-1, :, :] - Pdet_edges[1:, :, :]   # (n_q_bins, n_z, n_m)

    # numerical safety (should already be >=0, but keep tiny negatives from quadrature)
    Pbin = jnp.clip(Pbin, 0.0, 1.0)

    # ---- z-bin integration ----
    zlo = z_edges[:-1]
    zhi = z_edges[1:]

    def one_cell(z_l, z_h, P_q_mz):
        mask_z = ((z_grid >= z_l) & (z_grid < z_h))[:, None]  # (n_z, 1)
        integrand = dN_dzdlnM * P_q_mz * mask_z               # (n_z, n_m)

        partial_m = simpson(integrand, x=logm_grid, dx=dx_m, axis=1)  # (n_z,)
        val = simpson(partial_m, x=z_grid, dx=dx_z, axis=0)           # scalar
        return f_sky * val

    one_qbin = lambda P_q_mz: jax.vmap(lambda zl, zh: one_cell(zl, zh, P_q_mz))(zlo, zhi)  # (n_z_bins,)
    N_bins = jax.vmap(one_qbin)(Pbin)              # (n_q_bins, n_z_bins)
    N_bins = jnp.swapaxes(N_bins, 0, 1)            # (n_z_bins, n_q_bins)

    return N_bins, z_edges, q_edges

from .profiles import mpc_per_h_to_cm
import jax
import jax.numpy as jnp
from jax import lax
from . import classy_sz
from .utils import get_ell_range, simpson, get_ell_binwidth
from .tsz import get_integral_grid


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


# def compute_sigma_y0(M, z, params_values_dict, 
#                      sigma_obj_file = "/Users/licongxu/csd3/tsz_paper/noise_files/sigma_dict_szifi.npy", 
#                      skyfr_file = "/Users/licongxu/csd3/tsz_paper/noise_files/skyfracs_szifi_cosmology.npy",
#                      filter_name='immf6', theta_min=0.5, theta_max=32.0):
#     """
#     Compute sky-averaged y0 noise (sigma) at the cluster's theta_500.
#     Uses linear interpolation in log-theta and linear extrapolation
#     (via SciPy interp1d with fill_value='extrapolate').
#     """
#     # theta_500 (arcmin)
#     sigma_obj = jnp.load(sigma_obj_file, allow_pickle=True).item()  # {filter_name -> {tile_index -> 1D array over theta_500}}
#     skyfr = jnp.load(skyfr_file).ravel()  # shape (num_tiles,)
#     theta_500_arcmin = compute_theta500_arcmin(M, z, params_values_dict=params_values_dict)

#     # reconstruct theta grid and sky-avg curve
#     data = sigma_obj[filter_name]               # {tile_index -> 1D array over theta_500}
#     first = next(iter(data.values()))
#     ntheta = len(first)
#     theta_grid = jnp.exp(jnp.linspace(jnp.log(theta_min), jnp.log(theta_max), ntheta))

#     num = jnp.zeros(ntheta, dtype=float)
#     den = 0.0
#     for tile, arr in data.items():
#         w = float(skyfr[int(tile)])
#         y = jnp.asarray(arr, dtype=float)
#         num += w * y
#         den += w
#     sigma_skyavg = num / den  # shape (ntheta,)

#     # interpolator in log-theta with linear extrapolation
#     log_theta = jnp.log(theta_grid)
#     # build interpolator (note the trailing comma -> tuple of length 1)
#     interp = jax.scipy.interpolate.RegularGridInterpolator(
#         (log_theta,),            # <-- wrap in a tuple
#         sigma_skyavg,
#         method='linear',
#         bounds_error=False,
#         fill_value=None          # <-- enables extrapolation
#     )

#     # query: shape (n_points, ndim); for a single 1D point, that's (1, 1)
#     xi = jnp.array([[jnp.log(theta_500_arcmin)]])
#     sigma_at_theta500 = float(interp(xi)[0])

#     return sigma_at_theta500, float(theta_500_arcmin)


# def compute_sigma_y0(M, z, params_values_dict, 
#                      sigma_obj_file="/Users/licongxu/csd3/tsz_paper/noise_files/sigma_dict_szifi.npy", 
#                      skyfr_file="/Users/licongxu/csd3/tsz_paper/noise_files/skyfracs_szifi_cosmology.npy",
#                      filter_name='immf6', theta_min=0.5, theta_max=32.0):
#     """
#     Vectorized version.
#     Inputs:
#       M, z : scalars or 1D arrays with the same shape
#     Returns:
#       sigma_at_theta500 : same shape as inputs
#       theta_500_arcmin  : same shape as inputs
#     """

#     # Load sigma curves & sky fractions (once)
#     sigma_obj = jnp.load(sigma_obj_file, allow_pickle=True).item()  # {filter_name -> {tile_index -> 1D array over theta_500}}
#     skyfr = jnp.load(skyfr_file).ravel()                            # shape (num_tiles,)

#     # Reconstruct theta grid and sky-avg curve
#     data = sigma_obj[filter_name]               # {tile_index -> 1D array over theta_500}
#     first = next(iter(data.values()))
#     ntheta = len(first)
#     theta_grid = jnp.exp(jnp.linspace(jnp.log(theta_min), jnp.log(theta_max), ntheta))

#     num = jnp.zeros(ntheta, dtype=float)
#     # den = 0.0
#     den = jnp.array(0.0, dtype=float)
#     for tile, arr in data.items():
#         # w = float(skyfr[int(tile)])
#         w = skyfr[int(tile)]
#         y = jnp.asarray(arr, dtype=float)
#         num += w * y
#         den += w
#     sigma_skyavg = num / den  # shape (ntheta,)

#     # Build 1D interpolator in log-theta (supports batched queries)
#     log_theta = jnp.log(theta_grid)
#     interp = jax.scipy.interpolate.RegularGridInterpolator(
#         (log_theta,), 
#         sigma_skyavg,
#         method='linear',
#         bounds_error=False,
#         fill_value=None   # extrapolate
#     )

#     # Ensure array inputs; remember if inputs were scalar for output formatting
#     M_arr = jnp.atleast_1d(M)
#     z_arr = jnp.atleast_1d(z)
#     if M_arr.shape != z_arr.shape:
#         raise ValueError(f"M and z must have the same shape; got {M_arr.shape} vs {z_arr.shape}")

#     # Vectorized theta_500(M,z) using vmap around your scalar routine
#     theta_500_arcmin = jax.vmap(lambda m, zz: compute_theta500_arcmin(m, zz, params_values_dict))(M_arr, z_arr)

#     # Batched evaluation of the interpolator: xi must be shape (N, 1)
#     xi = jnp.log(theta_500_arcmin)[:, None]
#     sigma_at_theta500 = interp(xi)  # shape (N,)

#     # If inputs were scalars, return scalars; else return vectors
#     # if jnp.ndim(M) == 0 and jnp.ndim(z) == 0:
#     #     return float(sigma_at_theta500[0]), float(theta_500_arcmin[0])
#     # else:
#     #     return sigma_at_theta500, theta_500_arcmin
#     if jnp.ndim(M) == 0 and jnp.ndim(z) == 0:
#         return sigma_at_theta500[0], theta_500_arcmin[0]
#     else:
#         return sigma_at_theta500, theta_500_arcmin

def compute_sigma_y0(M, z, params_values_dict, 
                     sigma_obj_file="/Users/licongxu/csd3/tsz_paper/noise_files/sigma_dict_szifi.npy", 
                     skyfr_file="/Users/licongxu/csd3/tsz_paper/noise_files/skyfracs_szifi_cosmology.npy",
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
                sigma_obj_file="/Users/licongxu/csd3/tsz_paper/noise_files/sigma_dict_szifi.npy",
                skyfr_file="/Users/licongxu/csd3/tsz_paper/noise_files/skyfracs_szifi_cosmology.npy",
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
    sigma_obj_file="/Users/licongxu/csd3/tsz_paper/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/Users/licongxu/csd3/tsz_paper/noise_files/skyfracs_szifi_cosmology.npy",
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


@jax.jit
def compute_integral_snr_simple_uRC(
    params_values_dict=None,
    *,
    q_obs=None,
    sigma_obj_file="/Users/licongxu/csd3/tsz_paper/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/Users/licongxu/csd3/tsz_paper/noise_files/skyfracs_szifi_cosmology.npy",
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
    sigma_obj_file="/Users/licongxu/csd3/tsz_paper/noise_files/sigma_dict_szifi.npy",
    skyfr_file="/Users/licongxu/csd3/tsz_paper/noise_files/skyfracs_szifi_cosmology.npy",
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
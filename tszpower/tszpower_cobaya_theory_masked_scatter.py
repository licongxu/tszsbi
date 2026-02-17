#!/usr/bin/env python3
"""
Cobaya theory modules to compute the tSZ power spectrum WITH intrinsic scatter,
using the **parametric** pressure-profile amplitude:

    y0(M, z) = 10^{A_SZ} * (M/B / (0.7*3e14))^{alpha_SZ} * E(z)^2 * (h/0.7)^{-0.5}

Two Theory classes are provided:

1. ``tSZ_PS_Theory_Scatter_FullSky``
   Full-sky total tSZ PS.  No cluster masking.
   Scatter boosts the PS by exp(2 * sigma_lnY^2).

2. ``tSZ_PS_Theory_Scatter``
   Masked (unresolved) tSZ PS using double-scatter completeness.
   Scatter boosts the unresolved PS by exp(2 * sigma_lnY^2).
"""

from cobaya.theory import Theory
from .config import classy_sz
from .parametric_profile import (
    compute_tsz_Dell_binned_parametric,
    compute_masked_tsz_Dell_binned_parametric,
)
from .initialise import initialise
import numpy as np
import time


# =====================================================================
#  Shared constants
# =====================================================================

_FIXED_ASTRO = {
    "M_min": 1e14 * 0.6766,
    "M_max": 1e16 * 0.6766,
    "z_min": 5e-3,
    "z_max": 3.0,
    "P0GNFW": 8.130,
    "c500": 1.156,
    "gammaGNFW": 0.3292,
    "alphaGNFW": 1.0620,
    "betaGNFW": 5.4807,
    "jax": 1,
    "cosmo_model": 0,
}

_FIDUCIAL_COSMO = {
    "omega_b": 0.02242,
    "omega_cdm": 0.1193,
    "H0": 67.66,
    "tau_reio": 0.0544,
    "ln10^{10}A_s": 2.9718,
    "n_s": 0.9665,
    "B": 1.41,
}

_COMMON_PARAMS = {
    "omega_b": 0,
    "omega_cdm": 0,
    "H0": 0,
    "tau_reio": 0,
    "ln10_10A_s": 0,
    "n_s": 0,
    "B": 0,
    "A_SZ": 0,
    "alpha_SZ": 0,
    "sigma_lnY": 0,
}

_COMMON_REQS = {
    "omega_b": None,
    "omega_cdm": None,
    "H0": None,
    "tau_reio": None,
    "ln10_10A_s": None,
    "n_s": None,
    "B": None,
    "A_SZ": None,
    "alpha_SZ": None,
    "sigma_lnY": None,
}


def _build_pars(params_values, fixed_params):
    """Map cobaya param names -> internal names and merge with fixed params."""
    pars = {
        "omega_b": params_values["omega_b"],
        "omega_cdm": params_values["omega_cdm"],
        "H0": params_values["H0"],
        "tau_reio": params_values["tau_reio"],
        "ln10^{10}A_s": params_values["ln10_10A_s"],
        "n_s": params_values["n_s"],
        "B": params_values["B"],
    }
    pars.update(fixed_params)
    return pars


def _init_tszpower(fixed_params, fiducial_cosmo):
    """One-time heavy initialisation."""
    initial = dict(fixed_params)
    initial.update(fiducial_cosmo)
    classy_sz.set(initial)
    initialise()


###########################################################################
# 1.  Full-sky total tSZ PS  (parametric amplitude, scatter boost)
###########################################################################

class tSZ_PS_Theory_Scatter_FullSky(Theory):
    """
    Full-sky (total) tSZ power spectrum with parametric amplitude and
    intrinsic scatter.

    Theory prediction:
        D_ell = compute_tsz_Dell_binned_parametric(A_SZ, alpha_SZ)
                * exp(2 * sigma_lnY^2)

    The exp(2*sigma^2) factor arises because log-normal scatter in y0
    with width sigma_lnY boosts <y0^2> = <y0>^2 * exp(2*sigma_lnY^2).
    """

    output = ["Cl_sz"]
    params = dict(_COMMON_PARAMS)

    # sigma_lnY: float = 0.173

    def get_requirements(self):
        return dict(_COMMON_REQS)

    def initialize(self):
        self.fixed_params = dict(_FIXED_ASTRO)
        _init_tszpower(self.fixed_params, _FIDUCIAL_COSMO)

        self.log.info("tSZ_PS_Theory_Scatter_FullSky (parametric) initialized")

    def calculate(self, state, want_derived=True, **params_values):
        t0 = time.time()
        pars = _build_pars(params_values, self.fixed_params)

        A_SZ = float(params_values["A_SZ"])
        alpha_SZ = float(params_values["alpha_SZ"])
        sigma_lnY = float(params_values["sigma_lnY"])

        # Full-sky parametric PS (all halos, no masking)
        dl_param = compute_tsz_Dell_binned_parametric(
            params_values_dict=pars,
            A_SZ=A_SZ,
            alpha_SZ=alpha_SZ,
        )

        # Apply scatter boost
        scatter_boost = float(np.exp(2.0 * sigma_lnY ** 2))
        dl_scatter = np.asarray(dl_param) * scatter_boost

        state["Cl_sz"] = {"1h": dl_scatter, "2h": np.zeros_like(dl_scatter)}
        self._current_state = state

        self.log.info(
            "SZ PS (full-sky, parametric, scatter) computed in {:.4f}s".format(
                time.time() - t0
            )
        )

    def get_Cl_sz(self):
        return self._current_state.get("Cl_sz", None)


###########################################################################
# 2.  Masked (unresolved) tSZ PS  (parametric + double-scatter completeness)
###########################################################################

class tSZ_PS_Theory_Scatter(Theory):
    """
    Masked (unresolved) tSZ power spectrum with parametric amplitude
    and full double-scatter completeness.

    Theory prediction:
        D_ell = compute_masked_tsz_Dell_binned_parametric(
                    A_SZ, alpha_SZ, q_cat, sigma_lnY)
                * exp(2 * sigma_lnY^2)

    The parametric amplitude rescales both:
      - the PS integrand (ratio^2)
      - the SNR entering P_det (ratio)
    """

    output = ["Cl_sz"]
    params = dict(_COMMON_PARAMS)

    q_cat: float = 5.0
    sigma_lnY: float = 0.173

    sigma_obj_file: str = "/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy"
    skyfr_file: str = "/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy"
    filter_name: str = "immf6"
    theta_min: float = 0.5
    theta_max: float = 32.0

    n_grid: int = 1024
    nsig: float = 8.0

    def get_requirements(self):
        return dict(_COMMON_REQS)

    def initialize(self):
        self.fixed_params = dict(_FIXED_ASTRO)
        _init_tszpower(self.fixed_params, _FIDUCIAL_COSMO)

        self.log.info(
            "tSZ_PS_Theory_Scatter (parametric, masked) initialized | "
            f"q_cat={self.q_cat}"
        )

    def calculate(self, state, want_derived=True, **params_values):
        t0 = time.time()
        pars = _build_pars(params_values, self.fixed_params)

        A_SZ = float(params_values["A_SZ"])
        alpha_SZ = float(params_values["alpha_SZ"])
        sigma_lnY = float(params_values["sigma_lnY"])

        dl_total = compute_masked_tsz_Dell_binned_parametric(
            params_values_dict=pars,
            A_SZ=A_SZ,
            alpha_SZ=alpha_SZ,
            q_cat=self.q_cat,
            sigma_lnY=sigma_lnY,
            sigma_obj_file=self.sigma_obj_file,
            skyfr_file=self.skyfr_file,
            filter_name=self.filter_name,
            theta_min=self.theta_min,
            theta_max=self.theta_max,
            n_grid=self.n_grid,
            nsig=self.nsig,
            scale_1e12=True,
        )

        # Apply scatter boost
        scatter_boost = float(np.exp(2.0 * sigma_lnY ** 2))
        cl_1h = np.asarray(dl_total) * scatter_boost

        state["Cl_sz"] = {"1h": cl_1h, "2h": np.zeros_like(cl_1h)}
        self._current_state = state

        self.log.info(
            "SZ PS (masked, parametric, scatter) computed in {:.4f}s".format(
                time.time() - t0
            )
        )

    def get_Cl_sz(self):
        return self._current_state.get("Cl_sz", None)

#!/usr/bin/env python3
"""
Cobaya theory module to compute cluster number counts (CNC)
in (z, SNR) bins using the **parametric** pressure-profile amplitude
WITH intrinsic scatter (double-scatter completeness).

Parametric amplitude:
    y0(M, z) = 10^{A_SZ} * (M/B / (0.7*3e14))^{alpha_SZ}
                           * E(z)^2 * (h/0.7)^{-0.5}

The probability that a halo at (M, z) produces an observed SNR in
[q_lo, q_hi] is:
    P(q in [q_lo, q_hi] | M, z) = P_det(q_lo) - P_det(q_hi)

where P_det uses the parametric SNR (with rescaled y0) and the
full double-scatter completeness convolution.
"""

from cobaya.theory import Theory
import numpy as np
import jax.numpy as jnp
import time

from .config import classy_sz
from .initialise import initialise
from .parametric_profile import compute_cluster_counts_2d_parametric


class tSZ_CNC_Theory_Scatter(Theory):
    """
    Theory module that produces binned cluster number counts N_vec
    in (z, SNR) bins with parametric amplitude and scatter.

    Output:
        N2d_cnc : 2D numpy array, counts in all (z, SNR) bins.
    """

    output = ["N2d_cnc"]

    # Free (sampled) parameters
    params = {
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

    # ------------------------------------------------------------------
    # Configuration (overridable via YAML)
    # ------------------------------------------------------------------
    f_sky: float = 1.0
    sigma_lnY: float = 0.173

    n_z_bins: int = 10
    z_min: float = 5e-3
    z_max: float = 1.0

    n_snr_bins: int = 5
    snr_min: float = 5.0
    snr_max: float = 40.0

    sigma_obj_file: str = "/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy"
    skyfr_file: str = "/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy"
    filter_name: str = "immf6"
    theta_min: float = 0.5
    theta_max: float = 32.0

    n_grid: int = 1024
    nsig: float = 8.0

    # ------------------------------------------------------------------
    # Cobaya interface
    # ------------------------------------------------------------------

    def get_requirements(self):
        return {
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

    def initialize(self):
        self.log.info(
            "tSZ_CNC_Theory_Scatter (parametric) initialized | "
            f"sigma_lnY={self.sigma_lnY}"
        )

        self.fixed_params = {
            "M_min": 1e14 * 0.6766,
            "M_max": 1e16 * 0.6766,
            "z_min": max(self.z_min, 1e-6),
            "z_max": self.z_max,
            "P0GNFW": 8.130,
            "c500": 1.156,
            "gammaGNFW": 0.3292,
            "alphaGNFW": 1.0620,
            "betaGNFW": 5.4807,
            "jax": 1,
            "cosmo_model": 0,
        }

        initial_pars = dict(self.fixed_params)
        initial_pars.update({
            "omega_b": 0.02242,
            "omega_cdm": 0.1193,
            "H0": 67.66,
            "tau_reio": 0.0544,
            "ln10^{10}A_s": 2.972,
            "n_s": 0.9665,
            "B": 1.41,
        })

        classy_sz.set(initial_pars)
        initialise()
        self.log.info("Initial CNC precomputation completed.")

        self.z_bin_edges = np.linspace(self.z_min, self.z_max, self.n_z_bins + 1)
        self.snr_bin_edges = np.geomspace(
            self.snr_min, self.snr_max, self.n_snr_bins + 1
        )

        self.log.info(
            f"  z bins: {self.n_z_bins} [{self.z_min}, {self.z_max}]  |  "
            f"SNR bins: {self.n_snr_bins} [{self.snr_min}, {self.snr_max}] geom"
        )

        self._current_state = {}

    def calculate(self, state, want_derived=True, **params_values):
        t0 = time.time()

        pars = {
            "omega_b": params_values["omega_b"],
            "omega_cdm": params_values["omega_cdm"],
            "H0": params_values["H0"],
            "tau_reio": params_values["tau_reio"],
            "ln10^{10}A_s": params_values["ln10_10A_s"],
            "n_s": params_values["n_s"],
            "B": params_values["B"],
        }
        pars.update(self.fixed_params)

        A_SZ = float(params_values["A_SZ"])
        alpha_SZ = float(params_values["alpha_SZ"])

        # sigma_lnY: use sampled value if provided, else fall back to config
        sigma_lnY = float(params_values.get("sigma_lnY", self.sigma_lnY))

        N2d = compute_cluster_counts_2d_parametric(
            params_values_dict=pars,
            A_SZ=A_SZ,
            alpha_SZ=alpha_SZ,
            sigma_lnY=sigma_lnY,
            z_edges=jnp.array(self.z_bin_edges),
            q_edges=jnp.array(self.snr_bin_edges),
            f_sky=self.f_sky,
            sigma_obj_file=self.sigma_obj_file,
            skyfr_file=self.skyfr_file,
            filter_name=self.filter_name,
            theta_min=self.theta_min,
            theta_max=self.theta_max,
            n_grid=self.n_grid,
            nsig=self.nsig,
        )

        state["N2d_cnc"] = np.asarray(N2d)
        self._current_state = state

        self.log.info(
            "CNC (parametric, scatter) computed in {:.4f}s | "
            "sum(N) = {:.1f}".format(time.time() - t0, float(np.sum(N2d)))
        )

    def get_N2d_cnc(self):
        return self._current_state.get("N2d_cnc", None)

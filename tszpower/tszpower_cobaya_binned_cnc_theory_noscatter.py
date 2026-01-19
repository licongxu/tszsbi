#!/usr/bin/env python3
"""
Cobaya theory module to compute cluster number counts (CNC)
in (z, SNR) bins using the tszpower package.

This is analogous to the tSZ power spectrum theory module, but the
output is a flattened vector of binned cluster counts N_vec.
"""

from cobaya.theory import Theory
import numpy as np
import time

from .config import classy_sz
from .initialise import initialise
# Adjust this import to wherever you defined the function:
from .maskedpower import compute_cluster_counts_in_z_q_2d_bins_tophat


class tSZ_CNC_Theory(Theory):
    """
    Theory module that produces binned cluster number counts N_vec
    in (z, SNR) bins.

    Output:
        N_cnc : 1D numpy array, flattened counts in all (z, SNR) bins.
    """

    # Name of the output this theory provides
    output = ["N2d_cnc"]

    # Free (sampled) cosmological parameters (same as your PS theory)
    params = {
        "omega_b": 0,
        "omega_cdm": 0,
        "H0": 0,
        "tau_reio": 0,
        "ln10_10A_s": 0,
        "n_s": 0,
        "B": 0,
    }

    # ------------------------------------------------------------------
    # Configuration options that can be overridden in the Cobaya YAML
    # ------------------------------------------------------------------
    # survey sky fraction (full sky by default)
    f_sky: float = 1.0

    # Binning configuration
    n_z_bins: int = 10
    z_min: float = 0.
    z_max: float = 1.0

    n_snr_bins: int = 5
    snr_min: float = 5.0
    snr_max: float = 40.0

    # ------------------------------------------------------------------
    # Cobaya interface methods
    # ------------------------------------------------------------------

    def get_requirements(self):
        # We require the same cosmological parameters that we sample.
        return {
            "omega_b": None,
            "omega_cdm": None,
            "H0": None,
            "tau_reio": None,
            "ln10_10A_s": None,
            "n_s": None,
            "B": None,
        }

    def initialize(self):
        """
        Called once at the start. Set fixed astro parameters, binning,
        and run any heavy initialisation (e.g. CLASSy, interpolation tables).
        """
        self.log.info("tSZ_CNC_Theory initialized (cluster number counts).")

        # Fixed astrophysical & integration parameters
        # (taken from your allpars example)
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
            "cosmo_model": 0,  # 0: LCDM with fixed neutrino mass
        }

        # Build an initial parameter dictionary with fiducial values
        initial_pars = self.fixed_params.copy()
        initial_pars.update({
            "omega_b": 0.02242,
            "omega_cdm": 0.1193,
            "H0": 67.66,
            "tau_reio": 0.0544,
            # Key name expected by the underlying code:
            "ln10^{10}A_s": 2.972,
            "n_s": 0.9665,
            "B": 1.41,
        })

        # Set up CLASSy (or your cosmology backend) once
        classy_sz.set(initial_pars)
        initialise()  # your heavy precomputation (e.g. emulator setup, tables)
        self.log.info("Initial CNC precomputation completed.")

        # Precompute bin edges (can be overridden via YAML attributes)
        self.z_bin_edges = np.linspace(self.z_min, self.z_max, self.n_z_bins + 1)
        self.snr_bin_edges = np.geomspace(self.snr_min, self.snr_max, self.n_snr_bins + 1)



        # Prepare container for current state
        self._current_state = {}

    def calculate(self, state, want_derived=True, **params_values):
        """
        Called by Cobaya at each MCMC step with a new set of sampled parameters.
        It must fill 'state' with the requested outputs.
        """
        t0 = time.time()

        # Map sampled params (Cobaya names) to the names expected by your code.
        updated_pars = {
            "omega_b": params_values["omega_b"],
            "omega_cdm": params_values["omega_cdm"],
            "H0": params_values["H0"],
            "tau_reio": params_values["tau_reio"],
            # Rename ln10_10A_s -> ln10^{10}A_s
            "ln10^{10}A_s": params_values["ln10_10A_s"],
            "n_s": params_values["n_s"],
            "B": params_values["B"],
        }

        # Merge with fixed astrophysical + integration parameters
        updated_pars.update(self.fixed_params)

        # Compute binned cluster counts N_bins(z_i, snr_j)
        N2d = compute_cluster_counts_in_z_q_2d_bins_tophat(
            params_values_dict=updated_pars,
            z_edges=self.z_bin_edges,
            q_edges=self.snr_bin_edges,
            f_sky=self.f_sky,
        )  # shape (n_z_bins, n_snr_bins)

        state["N2d_cnc"] = np.asarray(N2d)  # keep as 2D array

        self._current_state = state

        self.log.info(
            "CNC (cluster counts) computed in {:.4f} seconds".format(time.time() - t0)
        )

    def get_N2d_cnc(self):
        return self._current_state.get("N2d_cnc", None)


#!/usr/bin/env python3
"""
Cluster Number Counts (CNC) Likelihood Module

Implements a *binned Poisson likelihood* for cluster number counts in
(z, SNR) bins.

It expects a theory module that provides a flattened vector N_cnc,
which is the predicted number of clusters in each (z, SNR) bin, in
the same binning and flattening convention as the data.

Data convention
---------------
We assume the data file contains a 2D array of observed counts H_data,
with shape (n_snr_bins, n_z_bins), constructed e.g. via:

    H, ln_snr_edges, z_edges = np.histogram2d(np.log(snr_sel), z_sel,
                                              bins=[ln_snr_bins, z_bins])
    # H.shape = (n_snr_bins, n_z_bins)

and then saved with np.savetxt. We then do:

    H_vec_data = H.flatten()  # row-major ordering

The theory vector N_cnc is constructed to match this ordering.
"""

import os
import numpy as np
from cobaya.likelihood import Likelihood


class tSZ_CNC_Likelihood(Likelihood):
    """
    Binned Poisson likelihood for cluster number counts in (z, SNR) bins.
    """

    # Configurable via YAML
    data_directory: str = "data"
    data_file: str = "data_cnc_counts_2d.txt"

    # Small floor for predicted counts to avoid log(0)
    lambda_floor: float = 1e-12

    # ------------------------------------------------------------------
    # Cobaya lifecycle methods
    # ------------------------------------------------------------------

    def initialize(self):
        """
        Load data and precompute any fixed quantities.
        """
        # Load the 2D histogram of observed counts
        data_path = os.path.join(self.data_directory, self.data_file)
        if not os.path.exists(data_path):
            raise RuntimeError(f"CNC data file not found: {data_path}")

        # Expecting a 2D array: shape (n_snr_bins, n_z_bins)
        H_data = np.loadtxt(data_path)

        # Flatten in the SAME way as your example:
        # H_vec = H.flatten()  # row-major order (C-order)
        self.N_obs_vec = H_data.flatten()
        self.n_bins = self.N_obs_vec.size

        self.log.info(
            f"tSZ_CNC_Likelihood: loaded data from {data_path}, "
            f"{H_data.shape} dimensions, "
            f"flattened length = {self.n_bins}"
        )

        super().initialize()

    def get_requirements(self):
        """
        Request the binned CNC prediction from the theory.
        The theory module should provide 'N_cnc' as a 1D vector of
        length matching self.N_obs_vec.
        """
        return {"N_cnc": {}}

    # ------------------------------------------------------------------
    # Poisson log-likelihood
    # ------------------------------------------------------------------

    def logp(self, **params_values):
        """
        Compute the Poisson log-likelihood:

            ln L = sum_i [ N_obs_i * ln(lambda_i) - lambda_i - ln(N_obs_i!) ]

        where lambda_i is the predicted mean count in bin i.

        The ln(N_obs_i!) term is independent of the theory and can be
        dropped (it just adds a constant offset to the log-likelihood),
        so we effectively use:

            ln L = sum_i [ N_obs_i * ln(lambda_i) - lambda_i ].

        To avoid log(0), we impose a small floor on lambda_i.
        """
        # 1) Get theory prediction from provider
        N_th_vec = self.provider.get_N_cnc()
        N_th_vec = np.asarray(N_th_vec, dtype=float)

        if N_th_vec.shape != self.N_obs_vec.shape:
            raise ValueError(
                f"Shape mismatch between theory N_cnc {N_th_vec.shape} "
                f"and data counts {self.N_obs_vec.shape}"
            )

        # 2) Apply floor to avoid log(0) and negative predictions
        lam = np.maximum(N_th_vec, self.lambda_floor)

        # 3) Poisson log-likelihood (dropping ln(N_obs!))
        N_obs = self.N_obs_vec
        logL = np.sum(N_obs * np.log(lam) - lam)

        # Optional: report simple diagnostics
        self.log.info(
            "tSZ_CNC_Likelihood: sum(N_obs) = %.3f, sum(N_th) = %.3f, loglike = %.3f",
            N_obs.sum(), N_th_vec.sum(), logL,
        )

        return logL

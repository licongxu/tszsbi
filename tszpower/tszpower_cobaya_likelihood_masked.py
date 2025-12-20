#!/usr/bin/env python3
"""
tSZ Power Spectrum Likelihood Module

This module implements a Gaussian likelihood for the tSZ power spectrum.
It reads the data from a text file and compares the theory prediction
(with multipole ℓ and power spectrum Cl) to the data.

This version uses a fixed covariance matrix loaded from file (no diagonal/other options).
"""

import os
import numpy as np
from cobaya.likelihood import Likelihood


class tSZ_PS_Likelihood(Likelihood):
    # Variables set via the YAML file.
    data_directory: str = "benchmark"
    data_file: str = "data_ell_yell_yerr_benchmark.txt"
    use_covariance: bool = False
    cov_file: str = "trispectrum_matrix_benchmark.txt"

    def initialize(self):
        # Load the data file.
        data_path = os.path.join(self.data_directory, self.data_file)
        D = np.loadtxt(data_path)
        # Expected format: Column 0: ℓ; Column 1: observed power spectrum; Column 2: error (sigma).
        self.ell_data = D[:, 0]
        self.cl_obs = D[:, 1]
        # self.sigma_obs = D[:, 2]  # kept for backward compatibility / logging, not used for cov building

        # --- Fixed covariance: must be loaded from file ---
        if self.cov_file is None:
            raise ValueError(
                "tSZ_PS_Likelihood: cov_file must be provided (fixed covariance mode)."
            )

        cov_path = os.path.join(self.data_directory, self.cov_file)
        C = np.loadtxt(cov_path)

        # Basic sanity checks and reshaping
        n = len(self.ell_data)
        C = np.atleast_2d(C)

        if C.shape != (n, n):
            raise ValueError(
                "tSZ_PS_Likelihood: loaded covariance has shape {} but expected ({}, {}). "
                "Make sure cov_file is an NxN matrix matching the number of data points.".format(
                    C.shape, n, n
                )
            )

        self.covmat = C

        # Pre-compute inverse and determinant.
        # Use slogdet for numerical stability.
        self.inv_covmat = np.linalg.inv(self.covmat)
        sign, logdet = np.linalg.slogdet(self.covmat)
        if sign <= 0:
            raise ValueError(
                "tSZ_PS_Likelihood: covariance matrix is not positive definite (slogdet sign <= 0)."
            )
        self.logdet_covmat = logdet

        self.log.info("tSZ_PS_Likelihood: Data loaded and fixed covariance matrix loaded.")
        super().initialize()

    def get_requirements(self):
        # Request the theory outputs from both SZ power and foreground components.
        return {"Cl_sz": {}, "Cl_sz_foreground": {}}

    def _get_data(self):
        return self.ell_data, self.cl_obs

    def _get_cov(self):
        return self.covmat

    def logp(self, **params_values):
        # Retrieve the SZ power spectrum from the theory.
        # Expecting Cl_sz to be a dictionary with keys "1h" and "2h"
        theory_dict = self.provider.get_Cl_sz()
        # Retrieve the foreground contribution.
        foreground = self.provider.get_Cl_sz_foreground()

        # Sum the SZ and foreground components.
        # cl_theory = theory_dict["1h"] + theory_dict["2h"] + foreground
        cl_theory = theory_dict["1h"] + foreground

        resid = self.cl_obs - cl_theory
        chi2 = np.dot(resid, np.dot(self.inv_covmat, resid))

        # If you want the normalized Gaussian likelihood, uncomment the next line
        # loglike = -0.5 * chi2 - 0.5 * self.logdet_covmat
        loglike = -0.5 * chi2

        self.log.info(
            "tSZ_PS_Likelihood: chi2 = {:.2f}, loglike = {:.2f}".format(chi2, loglike)
        )
        return loglike

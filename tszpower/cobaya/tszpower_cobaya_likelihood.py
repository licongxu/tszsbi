#!/usr/bin/env python3
"""
tSZ Power Spectrum Likelihood Module

This module implements a Gaussian likelihood for the tSZ power spectrum.
It reads the data from a text file and compares the theory prediction
(with multipole ℓ and C_ell) to the data.
"""

import os
import numpy as np
import jax.numpy as jnp
from cobaya.likelihood import Likelihood

class tSZ_PS_Likelihood(Likelihood):
    # These variables can be set via the YAML file.
    data_directory: str = "data"         # directory with the data file
    data_file: str = "data_ps-ell-y2-erry2_total-planck-collab-15.txt"                # file containing the data
    use_covariance: bool = False                   # if True, use provided covariance (if available)
    # (Optional) If you have an external covariance file, you could add:
    cov_file: str = None

    def initialize(self):
        # Read the data file.
        data_path = os.path.join(self.data_directory, self.data_file)
        D = np.loadtxt(data_path)
        # Expected file format:
        #   Column 0: ℓ values (multipole center)
        #   Column 1: Observed tSZ power spectrum (e.g., in units of [µK^2] or similar)
        #   Column 2: Gaussian error (sigma) on the measurement
        self.ell_data = D[:, 0]
        self.cl_obs = D[:, 1]
        self.sigma_obs = D[:, 2]

        # Build a diagonal covariance matrix if no external covariance is provided.
        # Otherwise, you could load an external file.
        if self.cov_file is None:
            self.covmat = np.diag(self.sigma_obs**2)
        else:
            cov_path = os.path.join(self.data_directory, self.cov_file)
            self.covmat = np.loadtxt(cov_path)

        # Pre-compute the inverse and determinant for the likelihood evaluation.
        self.inv_covmat = np.linalg.inv(self.covmat)
        self.det_covmat = np.linalg.det(self.covmat)
        self.log.info("tSZ_PS_Likelihood: Data loaded and covariance matrix constructed.")

        # super().initialize()

    def get_requirements(self):
        return {"tsz_power_spectrum": None, "ell": None}


    def logp(self, **params_values):
        """
        Compute the Gaussian log-likelihood:
          -0.5 * (data - theory)^T Cov^{-1} (data - theory)
          - 0.5 * ln(det(Cov))
        """
        # Retrieve the theory prediction from the provider.
        cl_theory = self.provider.get_tsz_power_spectrum()
        # cl = cl_theory["cl"]
        # ell = cl_theory["ell"]
        # Optionally, you might also retrieve the ℓ range from theory,
        # e.g., ell_theory = self.provider.get_ell(), and then interpolate
        # # the theory to the ℓ values of the data.
        # ell_theory = self.provider.get_ell()
        # For simplicity, assume that the theory spectrum is computed at the same ℓ
        # values as the data. Otherwise, you would interpolate here.
        # if len(ell_theory) != len(self.ell_data):
            # If the theory ℓ array is different, use numpy.interp to match the data points.
        #     cl_theory_interp = np.interp(self.ell_data, ell_theory, cl_theory)
        # else:
        #     cl_theory_interp = cl_theory

        # Compute the residual between the observed data and the theory.
        resid = self.cl_obs - cl_theory
        # Compute chi2: resid^T * inv_cov * resid
        chi2 = np.dot(resid, np.dot(self.inv_covmat, resid))
        # loglike = -0.5 * chi2 - 0.5 * np.log(self.det_covmat)
        loglike = -0.5 * chi2 
        self.log.info(f"tSZ_PS_Likelihood: chi2 = {chi2:.2f}, loglike = {loglike:.2f}")
        return loglike

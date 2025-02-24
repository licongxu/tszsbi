#!/usr/bin/env python3
"""
Cobaya theory module to compute the tSZ power spectrum using the tszpower package,
with fixed astrophysical parameters and free cosmological parameters,
plus additional foreground residuals.
"""

from cobaya.theory import Theory
import tszpower
import jax.numpy as jnp
import numpy as np
import os
import time

class tSZ_PS_Theory(Theory):
    # Declare the outputs.
    output = ["tsz_power_spectrum", "ell"]

    # --- Free (sampled) cosmological parameters ---
    omega_b: float         # Ω_b h²
    omega_cdm: float       # Ω_cdm h²
    H0: float              # Hubble parameter (km/s/Mpc)
    tau_reio: float        # reionization optical depth
    ln10_10A_s: float      # ln(10^10 A_s) amplitude (keep this parameter)
    n_s: float             # scalar spectral index
    B: float               # free parameter entering the pressure profile

    # --- Free nuisance (foreground) parameters ---
    A_sz: float            # overall amplitude for the tSZ power spectrum
    A_cib: float           # amplitude for the CIB foreground template
    A_ir: float            # amplitude for the infrared foreground template
    A_rs: float            # amplitude for the radio source foreground template

    # --- Additional options set via YAML ---
    foreground_template_directory: str = "data"
    foreground_template_file: str = "data_fg-ell-cib_rs_ir_cn-total-planck-collab-15.txt"

    def get_requirements(self):
        # List all free parameters.
        return {
            "omega_b": None,
            "omega_cdm": None,
            "H0": None,
            "tau_reio": None,
            "ln10_10A_s": None,
            "n_s": None,
            "B": None,
            "A_cib": None,
            "A_ir": None,
            "A_rs": None,
            "A_sz": None,
        }

    def initialize(self):
        self.log.info("tSZ_PS_Theory initialized with fixed astrophysical parameters and foreground templates.")
        self._current_state = {}

        # --- Fixed astrophysical parameters (and other fixed settings) ---
        # Cosmology is Planck 2018
        self.fixed_params = {
            # "omega_b":  0.0224, 
            # "n_s": 0.965,
            # "tau_reio": 0.054,
            "M_min": 1e11,
            "M_max": 5e15,
            # "ell_min": 10,
            # "ell_max": 959.5,
            # "dlogell": 0.114,
            "z_min": 1e-6,
            "z_max": 6.0,
            "P0GNFW": 8.130,
            "c500": 1.156,
            "gammaGNFW": 0.3292,
            "alphaGNFW": 1.0620,
            "betaGNFW": 5.4807,
            "jax": 1,
            # "cosmo_model": 1, 
        }

        # --- Load the foreground template data once during initialization ---
        fg_path = os.path.join(self.foreground_template_directory, self.foreground_template_file)
        if not os.path.exists(fg_path):
            raise RuntimeError(f"Foreground template file not found: {fg_path}")
        D_fg = np.loadtxt(fg_path)
        # Expected columns:
        #   Column 0: ell, Column 1: CIB template, Column 2: RS template,
        #   Column 3: IR template, Column 4: CN template.
        self.fg_ell = D_fg[:, 0]
        self.A_CIB_MODEL = D_fg[:, 1]
        self.A_RS_MODEL  = D_fg[:, 2]
        self.A_IR_MODEL  = D_fg[:, 3]
        self.A_CN_MODEL  = D_fg[:, 4]

        # # Perform a warm-up call to trigger any JIT compilation.
        # warmup_pars = self.fixed_params.copy()
        # warmup_pars.update({
        #     "omega_b": 0.02242,
        #     "omega_cdm": 0.1193,
        #     "H0": 67.66,
        #     "tau_reio": 0.0561,
        #     "ln10^{10}A_s": 3.047,t
        #     "n_s": 0.9665,
        #     "B": 1.0
        # })
        # tszpower.classy_sz.set(warmup_pars)
        # tszpower.classy_sz.compute_class_szfast()
        # _ = tszpower.compute_integral(params_values_dict=warmup_pars)
        # self._warmup_done = True
        # self.log.info("Warm-up call completed. JIT compilation (if any) has been triggered.")

    def calculate(self, state, want_derived=True, **params_values):
        # 1) Build the complete parameter dictionary.
        start_time = time.time()
        allpars = {
            "omega_b": params_values["omega_b"],
            "omega_cdm": params_values["omega_cdm"],
            "H0": params_values["H0"],
            "tau_reio": params_values["tau_reio"],
            "ln10^{10}A_s": params_values["ln10_10A_s"],
            "n_s": params_values["n_s"],
            "B": params_values["B"]
        }

        allpars.update(self.fixed_params)

        # 2) Setup tszpower.
        # print("Cobaya input parameters:", allpars)
        tszpower.classy_sz.set(allpars)
        # tszpower.warmup()
        tszpower.classy_sz.compute_class_szfast()
        
        ell = tszpower.get_ell_range()

        # 3) Compute the tSZ power spectrum.

        C_ell = tszpower.compute_integral(params_values_dict=allpars) * ell * (ell + 1) / (2 * jnp.pi) * 1e12
        # print("C_ell:", C_ell)
        

        # 4) Scale the tSZ power spectrum by the nuisance amplitude A_sz.
        tsz_component = params_values["A_sz"] * C_ell

        # 5) Compute the foreground contribution using pre-loaded template data.
        # Since the foreground templates have already been set to match the ℓ array,
        # we directly use them without interpolation.
        cib = params_values["A_cib"] * self.A_CIB_MODEL
        ir  = params_values["A_ir"]  * self.A_IR_MODEL
        rs  = params_values["A_rs"]  * self.A_RS_MODEL
        A_cn_fixed = 0.9033  # Fixed amplitude for the CN template
        cn  = A_cn_fixed * self.A_CN_MODEL

        foreground_component = cib + ir + rs + cn
        total_cl = tsz_component + foreground_component

        # 6) Store the results.
        self._current_state["tsz_power_spectrum"] = total_cl
        self._current_state["ell"] = ell

        state["tsz_power_spectrum"] = total_cl
        state["ell"] = ell

        if want_derived:
            state["derived"] = {}

        self.log.info("tSZ power spectrum (including foregrounds) computed with current parameters")
        end_time = time.time()
        print(f"compute_integral took {end_time - start_time:.4f} seconds")

    def get_tsz_power_spectrum(self):
        return self._current_state.get("tsz_power_spectrum", None)

    def get_ell(self):
        return self._current_state.get("ell", None)

# Plan: JAXifying CMB Lensing (p) for tszsbi and Accelerating cosmocnc Computations

This document describes:
1. How your tszsbi code and cosmocnc differ in their SNR (q) computation
2. How cosmocnc implements the lensing observable (p)
3. A concrete plan to implement lensing in tszsbi using JAX
4. Opportunities to accelerate cosmocnc-style computations with JAX

---

## Table of Contents

1. [SNR (q) Computation: tszsbi vs cosmocnc](#1-snr-q-computation-tszsbi-vs-cosmocnc)
2. [What cosmocnc Does for Lensing (p) That tszsbi Does Not](#2-what-cosmocnc-does-for-lensing-p-that-tszsbi-does-not)
3. [Plan: Implementing Lensing (p) in tszsbi with JAX](#3-plan-implementing-lensing-p-in-tszsbi-with-jax)
4. [JAX Acceleration Opportunities for cosmocnc-Style Computations](#4-jax-acceleration-opportunities-for-cosmocnc-style-computations)
5. [Summary of Recommended Approach](#5-summary-of-recommended-approach)

---

## 1. SNR (q) Computation: tszsbi vs cosmocnc

### 1.1 What they share

Both codes compute the mean tSZ signal-to-noise as:

$$\bar{q}(M,z) = \frac{y_0(M,z)}{\sigma_\mathrm{SZ}(\theta_{500}(M,z))}$$

Both use the same $\theta_{500}$ formula with the 6.997 prefactor. Both interpolate the noise curve $\sigma_\mathrm{SZ}(\theta_{500})$ using a polynomial fit in $\ln\theta$-vs-$\ln\sigma$ space (degree 3).

### 1.2 Key differences

| Aspect | **tszsbi** (`maskedpower.py`) | **cosmocnc** (`survey_sr_so_sim.py` + `cnc.py`) |
|--------|------|------|
| **Language / framework** | JAX (JIT-compiled, GPU-ready, autodiff) | NumPy + SciPy (CPU only) |
| **Signal $y_0$** | Full GNFW profile integral: computes $y_0$ from $P_0, c_{500}, \alpha, \beta, \gamma$ and the shape integral $\int_0^\infty x^{-\gamma}(1+x^\alpha)^{(\gamma-\beta)/\alpha}\,dx$ (hardcoded to 0.4705). Uses `classy_sz` for $r_{500}$, $H(z)$, etc. | Parametric: $\ln y_0 = \alpha_\mathrm{szifi}\,\ln M + \ln\!\bigl[10^A E(z)^2 (b_\mathrm{SZ} h_{70}/3)^\alpha / \sqrt{h_{70}}\bigr]$. A single power-law scaling relation. |
| **Noise curve** | Loaded from `sigma_dict_szifi.npy` (per-tile, sky-averaged), polynomial fit (deg 3) | Loaded from `so_sim_sz_mf_noise.npy` (pre-averaged), polynomial fit (deg 3) |
| **Scatter model** | Two layers: (1) intrinsic log-normal scatter $\sigma_{\ln Y}$ via `completeness_convolution_jax` — a 1D trapezoidal integral over $\ln q_m$ of $e^{-(t-\mu)^2/2\sigma^2}\,[1 - \mathrm{erf}((q_\mathrm{cat} - e^t)/\sqrt{2})]$; (2) instrumental noise implicitly included in the erfc selection function. Used for tSZ power spectrum masking and number count completeness. | Two layers: (1) intrinsic Gaussian scatter in $\ln q$ with variance $\sigma_{\ln q}^2$; (2) unit-variance Gaussian noise + $\chi$-like transformation $q_\mathrm{obs} = \sqrt{e^{2x_0}+\mathrm{dof}}$. Scatter is applied via FFT convolution of the HMF through the layer chain. Used for the unbinned number count likelihood. |
| **Cluster abundance** | Direct 2D integral: $N = \int dz\,d\ln M\;\frac{dN}{dz\,d\ln M}\times P_\mathrm{det}(M,z)$, with Simpson integration over $(z, \ln M)$ grids, `jax.vmap` over bins | Forward propagation: HMF is transformed through scaling relation layers via Jacobian variable transformation + FFT convolution with scatter at each layer. Produces a 2D abundance grid $\frac{d^2N}{dq\,dz}$. |
| **Per-cluster likelihood** | Not implemented (tszsbi computes aggregate quantities: power spectra, total/binned counts) | Backward convolutional approach: per-cluster integral $\mathcal{L}_i = \int d\ln M\;\frac{dn}{d\ln M}\,P(q_i,p_i\mid M,z_i)$ with FFT convolutions through layers |
| **Parallelisation** | `jax.vmap` over $(z, M)$ grid points, `lax.scan` over $\ell$ bins. Implicit GPU parallelism. | `multiprocess` (Python multiprocessing fork) over sky patches, redshift bins, or clusters. CPU only. |
| **Autodifferentiability** | Yes (JAX). Can compute gradients of number counts, power spectra w.r.t. parameters. | No. Parameters must be varied by re-evaluation (MCMC sampling). |

### 1.3 The fundamental algorithmic difference

**tszsbi** works in **(M, z) space**: it builds the detection probability $P_\mathrm{det}(M,z)$ on a $(z, M)$ grid and integrates the HMF against it. This is a **direct brute-force integral** approach. The SNR is a function of $(M, z)$, and the completeness/selection is applied as a weight on the $(z, M)$ grid.

**cosmocnc** works in **observable space**: it transforms the HMF from $\ln M$ into the observable $q$ (and then $q_\mathrm{obs}$) using a change of variable + FFT convolution at each layer. The result is the differential number counts $dN/dq\,dz$ directly on a grid in $(q, z)$ space. This is a **forward-propagation** approach.

For the **per-cluster unbinned likelihood** (which tszsbi does not implement), cosmocnc uses the **backward convolutional approach**: starting from the observed values, it propagates backwards through the layers using FFT convolutions to obtain $P(\boldsymbol{\omega}_\mathrm{obs} \mid M, z)$ as a function of $\ln M$, then integrates against the HMF.

### 1.4 Completeness function comparison

| Aspect | tszsbi | cosmocnc |
|--------|--------|----------|
| **Function** | `completeness_convolution_jax()` | Forward FFT convolution through layers in `get_cluster_abundance()` |
| **What it computes** | $P_\mathrm{det} = \frac{1}{2\sqrt{2\pi}\sigma}\int dt\,e^{-(t-\mu)^2/2\sigma^2}\,[1-\mathrm{erf}((q_\mathrm{cat}-e^t)/\sqrt{2})]$ | Convolves $dn/dx_1$ with Gaussian kernel at each layer using `scipy.signal.convolve(method="fft")`, then reads off the abundance above threshold |
| **Integration method** | `jnp.trapezoid` on a 1D grid in $\ln q$ | `scipy.signal.convolve` with FFT on a uniform grid in $x_1$ |
| **Speed** | Fast per-point (JIT-compiled), vmapped over $(z, M)$ grid | Fast for the whole HMF distribution at once (FFT is $O(n\log n)$) |
| **Result** | $P_\mathrm{det}(M,z)$ — a scalar per grid point | $dN/dq_\mathrm{obs}$ — the full distribution in observable space |

---

## 2. What cosmocnc Does for Lensing (p) That tszsbi Does Not

tszsbi currently has **no lensing observable**. cosmocnc includes `p_so_sim` (CMB lensing SNR) as a second mass observable. Here is what cosmocnc computes for p:

### 2.1 The lensing signal

From an NFW profile with fixed concentration $c = 3$:

$$\mathcal{F}_\mathrm{lens} = \frac{r_s\,\rho_0}{\Sigma_c} \times \kappa_\mathrm{NFW}(R=5c)$$

where:
- $r_s = \left(\frac{3}{4\rho_c\,500\,\pi\,c^3}\times 10^{15}\right)^{1/3}$
- $\rho_0 = \rho_c\frac{500}{3}\frac{c^3}{\ln(1+c)-c/(1+c)}$
- $\Sigma_c = \frac{D_\mathrm{CMB}}{4\pi\,D_A\,D_{l,\mathrm{CMB}}\,\gamma}$ (lensing critical surface density)
- $\kappa_\mathrm{NFW} = \frac{2(2-3R+R^3)}{3(-1+R^2)^{3/2}}$

### 2.2 The lensing noise

$\sigma_\mathrm{lens}(\theta_{500})$ interpolated from `so_sim_lensing_mf_noise.npy` via a log-log polynomial fit (degree 3).

### 2.3 The mean lensing SNR

$$\ln\bar{p}(M,z) = \ln\!\left[\mathcal{F}_\mathrm{lens}\,a_\mathrm{lens}\,(0.1\,b_\mathrm{cmblens})^{1/3}\right] + \frac{\ln M}{3} - \ln\sigma_\mathrm{lens}(\theta_{500}^\mathrm{lens})$$

### 2.4 The scatter model

- **Layer 0 (intrinsic):** $\ln p = \ln\bar{p} + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma_{\ln p}^2)$
- **Layer 1 (noise):** $p_\mathrm{obs} = e^{x_0}$ with unit-variance Gaussian noise in $\ln p$

### 2.5 How p enters the likelihood

- **Unbinned (per-cluster):** backward convolutional approach integrates $P(q_i, p_i \mid M, z_i)$ over $\ln M$
- **Stacked:** mean $\langle p \rangle$ computed from the mass posterior, compared to the observed stacked value

---

## 3. Plan: Implementing Lensing (p) in tszsbi with JAX

### 3.1 Step 1: Implement `compute_p_bar` (mean lensing SNR on a grid)

Create a JAX function analogous to `compute_snr()` but for lensing:

**Inputs:** $M$, $z$, cosmological parameters, lensing noise file  
**Output:** $\bar{p}(M, z)$

The function needs:

1. **NFW convergence signal** $\mathcal{F}_\mathrm{lens}(M, z)$:
   - Compute $\rho_c(z)$ from cosmology (already available via `classy_sz`)
   - Compute $D_A(z)$, $D_\mathrm{CMB}$, $D_{l,\mathrm{CMB}}$ from `classy_sz`
   - Hard-code $c = 3$ (or make it a parameter)
   - Compute $r_s$, $\rho_0$, $\Sigma_c$, $\kappa_\mathrm{NFW}$ as pure JAX arithmetic
   - The convergence factor $\kappa_\mathrm{NFW}(R=15) = \frac{2(2-45+3375)}{3(-1+225)^{3/2}}$ is a constant for $c=3$; can be precomputed

2. **Lensing angular scale** $\theta_{500}^\mathrm{lens}$:
   - Same formula as `compute_theta500_arcmin()` but with `bias_cmblens` instead of `B`
   - Can reuse the existing function with a different bias parameter

3. **Lensing noise** $\sigma_\mathrm{lens}(\theta_{500})$:
   - Load `so_sim_lensing_mf_noise.npy` (or equivalent noise file)
   - Polynomial fit in log-log (degree 3), same pattern as `compute_sigma_y0()`
   - Evaluate via JAX `polyval`

4. **Mean lensing SNR:**
   $$\bar{p} = \frac{\mathcal{F}_\mathrm{lens} \times a_\mathrm{lens} \times (0.1\,b_\mathrm{cmblens})^{1/3} \times M^{1/3}}{\sigma_\mathrm{lens}(\theta_{500}^\mathrm{lens})}$$

**Key JAX consideration:** All the intermediate quantities ($\rho_c$, $D_A$, etc.) must come from JAX-traceable functions. If `classy_sz` calls are not JAX-traceable, precompute them on the $(z)$ grid and interpolate inside JIT.

### 3.2 Step 2: Implement `build_p_grid` (lensing SNR on the (z, M) grid)

Analogous to `build_snr_grid()`:

```
def build_p_grid(m_grid, z_grid, params_values_dict, ...):
    # vmap over z, then vmap over M
    return jax.vmap(lambda z: jax.vmap(lambda m: compute_p_bar(m, z, ...))(m_grid))(z_grid)
```

Returns shape `(n_z, n_m)`.

### 3.3 Step 3: Implement scatter for lensing

For the **intrinsic scatter** in $\ln p$:

$$\ln p = \ln\bar{p} + \epsilon, \qquad \epsilon \sim \mathcal{N}(0, \sigma_{\ln p}^2)$$

This affects:
- **Number counts with lensing selection:** apply a completeness-like convolution in $\ln p$ space
- **Stacked lensing:** compute $\langle p \rangle(M,z) = \exp(\ln\bar{p} + \sigma_{\ln p}^2/2)$

For the stacked observable, the mean and variance at fixed $(M, z)$ are:

$$\langle p \rangle = e^{\ln\bar{p} + \sigma_{\ln p}^2/2}$$
$$\mathrm{Var}(p) = (e^{\sigma_{\ln p}^2}-1)\,e^{2\ln\bar{p}+\sigma_{\ln p}^2} + 1$$

These are simple JAX expressions.

### 3.4 Step 4: Use cases to implement

Depending on what you need, here are the lensing use cases ranked by complexity:

#### Use case A: Stacked lensing mean (simplest)

Given a set of clusters selected by $q > q_\mathrm{th}$, compute the expected stacked $\langle p \rangle$:

$$\langle p \rangle_\mathrm{stack} = \frac{\int dz\,d\ln M\;\frac{dN}{dz\,d\ln M}\,P_\mathrm{det}(M,z)\,\langle p\rangle(M,z)}{\int dz\,d\ln M\;\frac{dN}{dz\,d\ln M}\,P_\mathrm{det}(M,z)}$$

This is a straightforward 2D integral on your existing $(z, M)$ grid, weighted by $P_\mathrm{det}$ from the q-based completeness you already have.

#### Use case B: Binned lensing counts (moderate)

Bin clusters jointly in $(z, q, p)$ or compute $\langle p \rangle$ in $(z, q)$ bins. Requires building $\bar{p}(M,z)$ on the grid and integrating with appropriate bin masks.

#### Use case C: Joint (q, p) per-cluster unbinned likelihood (hardest)

This requires the backward convolutional approach or a direct integral for each cluster. See Section 4 for how to JAXify this.

### 3.5 Step 5: Cosmological distances for lensing

The lensing signal requires distances not needed for the tSZ:

| Quantity | Needed for | Source |
|----------|-----------|--------|
| $D_A(z)$ | $\theta_{500}$, $\Sigma_c$ | Already in `classy_sz` |
| $D_\mathrm{CMB}$ | $\Sigma_c$ | `classy_sz.get_angular_distance_at_z(z_CMB)` with $z_\mathrm{CMB} \approx 1089$ |
| $D_{l,\mathrm{CMB}}$ | $\Sigma_c$ | Luminosity distance to CMB, $= D_\mathrm{CMB}(1+z_\mathrm{CMB})^2$ if using angular diameter distance |
| $\rho_c(z)$ | NFW $\rho_0$ | From $\rho_c = 3H(z)^2/(8\pi G)$, available from `classy_sz` |

**Strategy:** Precompute these on the $z$ grid outside JIT, then interpolate inside JIT using `jnp.interp`.

---

## 4. JAX Acceleration Opportunities for cosmocnc-Style Computations

### 4.1 Bottleneck analysis of cosmocnc

| Component | Time complexity | cosmocnc approach | JAX opportunity |
|-----------|----------------|-------------------|-----------------|
| **HMF computation** | $O(n_z \times n_M)$ | Serial loop or `multiprocess` | `jax.vmap` over $z$, automatic GPU parallelism |
| **Abundance grid (forward convolution)** | $O(n_z \times n_\mathrm{patches} \times n_\mathrm{layers} \times n\log n)$ | `scipy.signal.convolve(method="fft")` in a Python loop | `jax.scipy.signal.convolve` or `jnp.fft` inside `vmap` over $z$ and patches |
| **Per-cluster backward convolution** | $O(N_\mathrm{clusters} \times n_{\ln M} \times n_\mathrm{layers} \times n\log n)$ | Python loop over clusters, `multiprocess` | `jax.vmap` over clusters. The backward convolution for each cluster is independent → embarrassingly parallel |
| **Stacked likelihood** | $O(N_\mathrm{stack} \times n_{\ln M})$ | Python loop over stacked clusters | `jax.vmap` trivially |
| **Scaling relation evaluation** | $O(n_\mathrm{grid})$ per call | NumPy vectorised | Already fast; JAX would add autodiff capability |

### 4.2 What to JAXify (high impact)

#### A. The abundance grid (forward FFT convolution)

The inner loop in `get_cluster_abundance()` iterates over $(z, \mathrm{patch})$ and for each calls:
1. `eval_scaling_relation` → pure arithmetic, trivially JAX-able
2. `eval_derivative_scaling_relation` → pure arithmetic; with JAX, you get this **for free** via `jax.grad`
3. `convolve_1d` → `scipy.signal.convolve(method="fft")`

**JAX replacement:**
- Replace the Jacobian computation with `jax.jacfwd` or `jax.grad` of the scaling relation
- Replace `scipy.signal.convolve` with `jax.scipy.signal.convolve` or a manual FFT approach: `jnp.fft.fft` → multiply → `jnp.fft.ifft`
- `vmap` over all $(z, \mathrm{patch})$ combinations simultaneously

**Expected speedup:** 10–100x on GPU (the current Python loop over $n_z \times n_\mathrm{patches}$ is the main bottleneck; vmap eliminates it entirely).

#### B. The backward convolution (per-cluster likelihood)

Each cluster's likelihood is independent. The current code loops over clusters (parallelised via `multiprocess`). With JAX:

- Write a single-cluster backward convolution function
- `vmap` over all clusters
- On GPU, this runs all ~16,000 clusters in parallel

**Expected speedup:** 50–500x on GPU (eliminating Python loop overhead + GPU parallelism).

**Challenge:** The adaptive mass range (different $\ln M$ grid per cluster) breaks `vmap`'s requirement for uniform array shapes. Solutions:
1. **Pad to a common grid size** and mask out-of-range values
2. **Use a fixed, sufficiently wide grid** for all clusters (trades some memory for vmappability)
3. **Batch clusters by similar mass range** and vmap within each batch

#### C. The completeness convolution itself

Your tszsbi `completeness_convolution_jax` already does this in JAX! It computes $P_\mathrm{det}$ via trapezoidal integration. This is equivalent to cosmocnc's forward FFT convolution but from a different angle (direct integral vs. distribution convolution).

For the lensing use case, you can write an analogous completeness function for $p$ if needed, or use the simpler stacked-mean approach.

### 4.3 What NOT to JAXify (low impact / difficult)

| Component | Reason |
|-----------|--------|
| `classy_sz` / CLASS calls | C/Fortran backend, not JAX-traceable. Precompute on grid and interpolate. |
| File I/O (noise curves, catalogue loading) | One-time cost, negligible |
| MCMC sampler (Cobaya) | External tool; JAX benefit is in the likelihood evaluation speed |

### 4.4 The autodiff advantage

With JAX, you get **automatic differentiation** of the likelihood with respect to all parameters. This enables:
- **Gradient-based sampling** (HMC, NUTS) instead of Metropolis-Hastings → much faster convergence for high-dimensional parameter spaces
- **Fisher matrix forecasts** via `jax.hessian` of the log-likelihood
- **Sensitivity analysis** via `jax.grad` of number counts w.r.t. cosmological parameters

---

## 5. Summary of Recommended Approach

### Phase 1: Add lensing signal to tszsbi (no likelihood changes)

1. Implement `compute_lensing_signal(M, z, ...)` — NFW convergence, pure JAX arithmetic
2. Implement `compute_sigma_lens(M, z, ...)` — polynomial fit to lensing noise, same pattern as `compute_sigma_y0`
3. Implement `compute_p_bar(M, z, ...)` = signal / noise
4. Implement `build_p_grid(m_grid, z_grid, ...)` via `jax.vmap`
5. Implement `compute_stacked_p(...)` — weighted average of $\langle p \rangle(M,z)$ over detected clusters

### Phase 2: Use lensing in aggregate observables

6. Compute expected $\langle p \rangle$ in $(z, q)$ bins alongside your existing binned counts
7. Add lensing-weighted tSZ power spectrum (weight integrand by $p$ or mask by $p$ threshold)

### Phase 3: Per-cluster unbinned likelihood (if needed)

8. Implement a JAX version of the backward convolutional approach
9. Key function: `single_cluster_loglik(q_obs, p_obs, z, lnM_grid, hmf, scaling_rels, scatter_covs)`
10. `vmap` over all clusters for the full catalogue likelihood
11. Wire up to a gradient-based sampler (e.g., `numpyro`, `blackjax`, or JAX-compatible Cobaya)

### Key design principle

Keep the **precomputation** (cosmological distances, HMF on grid, noise curves) **outside JIT**, and make the **likelihood evaluation** (scaling relations, scatter convolution, integration over mass) **inside JIT**. This mirrors what you already do in tszsbi for the tSZ power spectrum.

### File structure suggestion

```
tszsbi/tszpower/
├── maskedpower.py          # existing: tSZ power spectrum, SNR, number counts
├── lensing.py              # NEW: compute_lensing_signal, compute_p_bar, build_p_grid
├── lensing_likelihood.py   # NEW (Phase 3): backward convolution, per-cluster loglik
├── profiles.py             # existing
├── tsz.py                  # existing
└── utils.py                # existing
```

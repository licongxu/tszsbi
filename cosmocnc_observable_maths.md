# cosmocnc: Mathematical Formalism for the tSZ SNR (q) and CMB Lensing SNR (p) Observables

Reference: [Zubeldia & Bolliet (2024), arXiv:2403.09589](https://arxiv.org/abs/2403.09589)

Code: `/scratch/scratch-lxu/cosmocnc/`  
Survey file (SO-like): `surveys/survey_sr_so_sim.py`

---

## Table of Contents

1. [General Hierarchical Model](#1-general-hierarchical-model)
2. [Case 1: SNR-Only (q) — Selection Observable](#2-case-1-snr-only-q--selection-observable)
3. [Case 2: SNR + CMB Lensing (q + p) — Two Mass Observables](#3-case-2-snr--cmb-lensing-q--p--two-mass-observables)
4. [Likelihood Structure](#4-likelihood-structure)
5. [Stacked Lensing Observable](#5-stacked-lensing-observable)
6. [Backend Algorithm Implementation Details](#6-backend-algorithm-implementation-details)
7. [Code-to-Maths Mapping](#7-code-to-maths-mapping)

---

## 1. General Hierarchical Model

cosmocnc relates observed mass proxies $\boldsymbol{\omega}_\mathrm{obs}$ to the true cluster mass $M$ and redshift $z$ through a hierarchical model with $n_\mathrm{layer}$ layers.

Each layer $j$ performs two operations:

1. **Scaling relation** (deterministic):

$$\boldsymbol{\omega}_{\mathrm{in}}^{(j)} = \mathbf{f}^{(j)}\!\left(\boldsymbol{\omega}^{(j-1)},\, z,\, \hat{\mathbf{n}}\right)$$

2. **Gaussian scatter** (stochastic):

$$\boldsymbol{\omega}^{(j)} = \boldsymbol{\omega}_{\mathrm{in}}^{(j)} + \boldsymbol{\epsilon}^{(j)}, \qquad \boldsymbol{\epsilon}^{(j)} \sim \mathcal{N}\!\left(\mathbf{0},\, \mathbf{C}^{(j)}\right)$$

The boundary conditions are:

- Input to layer 0: $\omega_k^{(0)} = \ln(M / 10^{14}\,M_\odot)$ for all observables $k$.
- Output of the last layer: $\boldsymbol{\omega}^{(n_\mathrm{layer})} = \boldsymbol{\omega}_\mathrm{obs}$ (the measured values).

For the SO-like catalogue, **both q and p use 2 layers** ($n_\mathrm{layer} = 2$).

---

## 2. Case 1: SNR-Only (q) — Selection Observable

### Configuration

```python
cnc_params["observables"] = [["q_so_sim"]]
cnc_params["obs_select"]  = "q_so_sim"
```

Only one mass observable: the tSZ signal-to-noise $q$.

### 2.1 Layer 0: Mass → ln(mean SNR)

**Input:** $x_0 = \ln(M / 10^{14}\,M_\odot)$

**Scaling relation:**

$$\ln \bar{q}(M,z) = \ln y_0(M,z) - \ln \sigma_\mathrm{SZ}\!\bigl(\theta_{500}(M,z)\bigr)$$

where:

- **Compton-$y$ amplitude:**

$$\ln y_0 = \alpha\, x_0 + \ln\!\left[10^{A}\, E(z)^2 \left(\frac{b_\mathrm{SZ}}{3}\,h_{70}\right)^{\!\alpha} h_{70}^{-1/2}\right]$$

  with $A = A_\mathrm{szifi}$, $\alpha = \alpha_\mathrm{szifi}$, $b_\mathrm{SZ} = $ `bias_sz` (hydrostatic mass bias, i.e. $1-b$), and $h_{70} = H_0/70$.

- **Angular scale:**

$$\theta_{500} = 6.997 \left(\frac{H_0}{70}\right)^{\!-2/3} \left(\frac{b_\mathrm{SZ}}{3}\right)^{\!1/3} E(z)^{-2/3} \frac{500}{D_A(z)} \times M^{1/3}$$

- **SZ noise** $\sigma_\mathrm{SZ}(\theta_{500})$: interpolated from a precomputed noise curve (`so_sim_sz_mf_noise.npy`) via a polynomial fit in $\ln\theta_{500}$ vs $\ln\sigma_\mathrm{SZ}$ space (degree 3).

**Scatter:** Gaussian in $\ln q$ with variance $\sigma_{\ln q}^2$:

$$\ln q^{(0)} = \ln\bar{q} + \epsilon_{\ln q}, \qquad \epsilon_{\ln q} \sim \mathcal{N}(0,\, \sigma_{\ln q}^2)$$

Code: `sigma_lnq_szifi` (default 0.173).

### 2.2 Layer 1: Measurement noise

**Scaling relation:**

$$q_\mathrm{obs} = \sqrt{e^{2\,x_0} + \mathrm{dof}}$$

where $x_0$ is the output of layer 0 (after scatter), and `dof` is a degrees-of-freedom parameter (default 0). When $\mathrm{dof} = 0$, this reduces to $q_\mathrm{obs} = e^{x_0}$.

**Scatter:** unit-variance Gaussian noise ($\sigma^2 = 1$) applied to $x_0$ before the above transformation.

**Selection cut:** $q_\mathrm{obs} > q_\mathrm{th}$ (default `obs_select_min` = 6).

### 2.3 Cluster abundance computation

The predicted differential number counts $\frac{dN}{dq\,dz}$ are computed by:

1. Start with the halo mass function $\frac{dn}{d\ln M}(M, z)$.
2. For each layer $k = 0, 1$:
   - Apply the variable transformation: $\frac{dn}{dx_1} = \frac{dn}{dx_0} / \left|\frac{dx_1}{dx_0}\right|$
   - Convolve with Gaussian scatter (FFT convolution): $\frac{dn}{dx_1} \to \frac{dn}{dx_1} * \mathcal{N}(0, \sigma_k^2)$
   - Re-interpolate onto a uniform grid.
3. The final $\frac{dn}{dq_\mathrm{obs}}$ is evaluated on `obs_select_vec` and multiplied by $4\pi f_\mathrm{sky}$.

### 2.4 Unbinned likelihood (q only)

When only the selection observable is available, the per-cluster likelihood is read directly from the abundance:

$$\ln \mathcal{L}_\mathrm{data} = \sum_{i} \ln\!\left[\frac{d^2 N}{dq\,dz}\bigg|_{q_i, z_i}\right]$$

The total log-likelihood also includes the Poisson term:

$$\ln \mathcal{L} = -N_\mathrm{pred} + \sum_i \ln\!\left[\frac{d^2 N}{dq\,dz}\bigg|_{q_i, z_i}\right]$$

where $N_\mathrm{pred} = \int dq\,dz\, \frac{d^2N}{dq\,dz}$.

---

## 3. Case 2: SNR + CMB Lensing (q + p) — Two Mass Observables

### Configuration

```python
cnc_params["observables"] = [["q_so_sim"], ["p_so_sim"]]
cnc_params["obs_select"]  = "q_so_sim"
```

Two mass observables: tSZ SNR $q$ (selection) and CMB lensing SNR $p$ (additional).

The observables are placed in **separate correlation sets** (each in its own inner list), meaning their intrinsic scatters ($\epsilon_{\ln q}$, $\epsilon_{\ln p}$) are **uncorrelated** at layer 0 (when `corr_lnq_lnp` = 0). If they were in the same inner list `[["q_so_sim", "p_so_sim"]]`, their scatter would be jointly correlated.

### 3.1 The q observable (same as Case 1)

Identical to Section 2 above. Layer 0 gives $\ln\bar{q}(M,z)$; layer 1 gives $q_\mathrm{obs}$.

### 3.2 The p observable: Layer 0 (Mass → ln(mean lensing SNR))

**Input:** $x_0 = \ln(M / 10^{14}\,M_\odot)$

**Scaling relation:**

$$\ln \bar{p}(M,z) = \ln\!\left[\mathcal{F}_\mathrm{lens}\, a_\mathrm{lens}\, (0.1\,b_\mathrm{cmblens})^{1/3}\right] + \frac{x_0}{3} - \ln\sigma_\mathrm{lens}\!\bigl(\theta_{500}^\mathrm{lens}\bigr)$$

where the terms are:

#### NFW convergence prefactor $\mathcal{F}_\mathrm{lens}$

An NFW profile with concentration $c = 3$ is assumed:

$$r_s = \left(\frac{3}{4\,\rho_c\, 500\,\pi\, c^3}\times 10^{15}\right)^{1/3}$$

$$\rho_0 = \rho_c \frac{500}{3} \frac{c^3}{\ln(1+c) - c/(1+c)}$$

$$\Sigma_c = \frac{D_\mathrm{CMB}}{4\pi\, D_A\, D_{l,\mathrm{CMB}}\, \gamma}$$

$$R = 5c, \qquad \kappa_\mathrm{NFW} = \frac{2(2 - 3R + R^3)}{3(-1 + R^2)^{3/2}}$$

$$\mathcal{F}_\mathrm{lens} = \frac{r_s\, \rho_0}{\Sigma_c} \times \kappa_\mathrm{NFW}$$

Here $\rho_c$ is the critical density at redshift $z$, $D_A$ is the angular diameter distance to the cluster, $D_\mathrm{CMB}$ to the CMB, $D_{l,\mathrm{CMB}}$ is the luminosity distance to the CMB, and $\gamma = 4G/c^2$ in appropriate units.

#### Lensing angular scale

$$\theta_{500}^\mathrm{lens} = 6.997 \left(\frac{H_0}{70}\right)^{\!-2/3} \left(\frac{b_\mathrm{cmblens}}{3}\right)^{\!1/3} E(z)^{-2/3} \frac{500}{D_A} \times M^{1/3}$$

(same form as the SZ angular scale but with $b_\mathrm{cmblens}$ instead of $b_\mathrm{SZ}$).

#### Lensing noise

$\sigma_\mathrm{lens}(\theta_{500}^\mathrm{lens})$: interpolated from `so_sim_lensing_mf_noise.npy` via a polynomial fit in $\ln\theta_{500}^\mathrm{lens}$ vs $\ln\sigma_\mathrm{lens}$ space (degree 3).

#### Physical meaning

The mean lensing SNR is the ratio of the theoretical lensing convergence signal (from the NFW profile) to the lensing reconstruction noise at the cluster's angular scale:

$$\bar{p} \propto \frac{\kappa_\mathrm{signal}(M,z)}{\sigma_\mathrm{lens}(\theta_{500})}$$

**Scatter:** Gaussian in $\ln p$ with variance $\sigma_{\ln p}^2$:

$$\ln p^{(0)} = \ln\bar{p} + \epsilon_{\ln p}, \qquad \epsilon_{\ln p} \sim \mathcal{N}(0,\, \sigma_{\ln p}^2)$$

Code: `sigma_lnp` (default 0.22).

### 3.3 The p observable: Layer 1 (Observational noise)

**Scaling relation:**

$$p_\mathrm{obs} = e^{x_0}$$

where $x_0$ is the output of layer 0 (after scatter).

**Scatter:** unit-variance Gaussian noise ($\sigma^2 = 1$) in $\ln p$ space.

So the full chain is:

$$p_\mathrm{obs} = \exp\!\bigl(\ln\bar{p} + \epsilon_{\ln p} + \eta\bigr), \qquad \eta \sim \mathcal{N}(0, 1)$$

### 3.4 Key difference between q and p at Layer 1

| Observable | Layer 1 transformation | Layer 1 scatter |
|:----------:|:----------------------:|:---------------:|
| $q$ (tSZ SNR) | $q_\mathrm{obs} = \sqrt{e^{2x_0} + \mathrm{dof}}$ | $\sigma^2 = 1$ |
| $p$ (lensing SNR) | $p_\mathrm{obs} = e^{x_0}$ | $\sigma^2 = 1$ |

The tSZ SNR uses a $\chi$-like transformation (reflecting matched-filter statistics), while the lensing SNR uses a simple exponentiation (log-normal noise model).

### 3.5 Full scatter covariance structure

#### Layer 0 (intrinsic scatter):

$$\mathbf{C}^{(0)} = \begin{pmatrix} \sigma_{\ln q}^2 & \rho_{\ln q,\ln p}\,\sigma_{\ln q}\,\sigma_{\ln p} \\ \rho_{\ln q,\ln p}\,\sigma_{\ln q}\,\sigma_{\ln p} & \sigma_{\ln p}^2 \end{pmatrix}$$

Default values: $\sigma_{\ln q} = 0.173$, $\sigma_{\ln p} = 0.22$, $\rho_{\ln q,\ln p} = 0.77$.

#### Layer 1 (measurement noise):

$$\mathbf{C}^{(1)} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

Measurement noise is **uncorrelated** between q and p at layer 1.

### 3.6 Unbinned likelihood with q + p (backward convolutional approach)

When both $q$ and $p$ are available for a cluster, the per-cluster contribution to the likelihood is **not** taken from the abundance grid. Instead, it is evaluated via the **backward convolutional approach** by integrating over $\ln M$:

$$\mathcal{L}_i = \int d\ln M\; \frac{dn}{d\ln M}(M, z_i) \times P(q_i, p_i \mid M, z_i)$$

The conditional pdf $P(q_i, p_i \mid M, z_i)$ is computed by propagating through the 2-layer hierarchy:

1. Evaluate the scaling relations at each layer for a grid of $\ln M$ values.
2. At the outermost layer, evaluate the Gaussian pdf of the residual between observed and predicted values.
3. Convolve backwards through intermediate layers using FFT.

Since q and p are in **separate correlation sets** (uncorrelated intrinsic scatter, `corr_lnq_lnp` = 0), the conditional pdf factorises:

$$P(q_i, p_i \mid M, z_i) = P(q_i \mid M, z_i) \times P(p_i \mid M, z_i)$$

and each factor is convolved independently, yielding a significant speedup.

When `corr_lnq_lnp` $\neq 0$, they must be placed in the **same** correlation set and evaluated jointly (slower but accounts for correlated scatter).

The total unbinned log-likelihood remains:

$$\ln \mathcal{L} = -N_\mathrm{pred} + \sum_i \ln \mathcal{L}_i$$

where $N_\mathrm{pred}$ comes from the cluster abundance integrated over $q > q_\mathrm{th}$ and $z$ (computed from q alone, since q is the selection observable).

---

## 4. Likelihood Structure

### 4.1 Poisson term (same for both cases)

$$\ln \mathcal{L}_\mathrm{Poisson} = -N_\mathrm{pred}$$

where $N_\mathrm{pred} = \sum_j \int_{q_\mathrm{th}}^{q_\mathrm{max}} dq \int_{z_\mathrm{min}}^{z_\mathrm{max}} dz\; \frac{d^2N}{dq\,dz}\bigg|_{\mathrm{patch}\,j}$

This is computed entirely from q (the selection observable) and the halo mass function.

### 4.2 Data term

| Scenario | Data term per cluster |
|----------|----------------------|
| q only | $\ln\frac{d^2N}{dq\,dz}\bigg\vert_{q_i,z_i}$ (read from abundance grid) |
| q + p (both available) | $\ln \int d\ln M\; \frac{dn}{d\ln M}\, P(q_i, p_i \mid M, z_i)$ (backward convolution) |
| q + p (only q available for cluster $i$) | $\ln\frac{d^2N}{dq\,dz}\bigg\vert_{q_i,z_i}$ (read from abundance grid, same as q-only) |

---

## 5. Stacked Lensing Observable

When `stacked_likelihood = True`, cosmocnc additionally computes the expected mean lensing SNR:

$$\langle p \rangle = \frac{1}{N_\mathrm{stack}} \sum_{i=1}^{N_\mathrm{stack}} \int d\ln M\; P(\ln M \mid q_i, z_i)\; \langle p \rangle(M, z_i)$$

where the mean of $p$ at fixed $\ln M$ accounts for the log-normal intrinsic scatter:

$$\langle p \rangle(M, z) = \exp\!\left(\ln\bar{p}(M,z) + \frac{\sigma_{\ln p}^2}{2}\right)$$

and the variance:

$$\mathrm{Var}(p \mid M, z) = \left(e^{\sigma_{\ln p}^2} - 1\right) e^{2\ln\bar{p} + \sigma_{\ln p}^2} + 1$$

The "$+1$" accounts for unit-variance measurement noise at layer 1.

The stacked log-likelihood is:

$$\ln \mathcal{L}_\mathrm{stacked} = -\frac{1}{2} \frac{(\langle p \rangle_\mathrm{obs} - \langle p \rangle_\mathrm{pred})^2}{\mathrm{Var}_\mathrm{pred}}$$

---

## 6. Backend Algorithm Implementation Details

This section describes the actual numerical algorithms used in the code, tracing the call flow through the source files.

### 6.1 Abundance grid: forward layer-by-layer FFT convolution

**Entry point:** `cnc.py` → `get_cluster_abundance()` (line ~265)

This is used for **q only** (the selection observable). The abundance grid $\frac{d^2N}{dq\,dz}$ is precomputed on a 2D grid of `(z, q)` and stored in `abundance_tensor[patch, z_index, q_index]`.

**Algorithm (per redshift bin, per sky patch):**

1. **Start with the HMF on a ln M grid:**
   ```
   x0 = self.ln_M                         # shape: (n_points,)
   dn_dx0 = self.hmf_matrix[z_index, :]   # dn/d(lnM) at this redshift
   ```

2. **Loop over layers** (`k = 0, 1`):

   a. **Evaluate the deterministic scaling relation** $x_1 = f^{(k)}(x_0)$:
      ```python
      x1 = scal_rel_selection.eval_scaling_relation(x0, layer=k, ...)
      ```
   
   b. **Compute the Jacobian** $dx_1/dx_0$ (analytically or numerically):
      ```python
      dx1_dx0 = scal_rel_selection.eval_derivative_scaling_relation(x0, layer=k, ...)
      ```
   
   c. **Variable transformation** — change the distribution from $x_0$-space to $x_1$-space:
      ```python
      dn_dx1 = dn_dx0 / dx1_dx0
      ```
   
   d. **Re-interpolate onto a uniform grid** in $x_1$:
      ```python
      x1_interp = np.linspace(min(x1), max(x1), n_points)
      dn_dx1 = np.interp(x1_interp, x1, dn_dx1)
      ```
   
   e. **Convolve with Gaussian scatter** using `scipy.signal.convolve` with `method="fft"`:
      ```python
      # In utils.py, convolve_1d():
      kernel = gaussian_1d(x - mean(x) + dx/2, sigma)
      dn_dx1 = signal.convolve(dn_dx1, kernel, mode="same", method="fft") / sum(kernel)
      ```
      The convolution method is controlled by `abundance_integral_type` (default `"fft"`). This uses `scipy.signal.convolve(..., method="fft")`, which internally performs:
      - FFT of the distribution
      - FFT of the Gaussian kernel
      - Pointwise multiplication in Fourier space
      - Inverse FFT
      
      Alternatively, `method="direct"` performs a brute-force direct convolution.
   
   f. **Pass to next layer**: `x0 = x1_interp`, `dn_dx0 = dn_dx1`.

3. **Final interpolation** onto the output observable grid `obs_select_vec`:
   ```python
   dn_dx1_interp = np.interp(self.obs_select_vec, x0, dn_dx0)
   abundance = dn_dx1_interp * 4 * pi * f_sky[patch]
   ```

4. **Integration** to get $N_\mathrm{pred}$:
   ```python
   n_obs_matrix[patch, :] = simpson(abundance_tensor[patch, :, :], x=redshift_vec, axis=0)
   n_tot_vec[patch] = simpson(n_obs_matrix[patch, :], x=obs_select_vec)
   ```

**Grid sizes:** `n_points` (default 4096) for the observable axis, `n_z` (default 50) for redshift. The large `n_points` ensures adequate FFT resolution.

### 6.2 Per-cluster likelihood: backward convolutional approach

**Entry point:** `cnc.py` → `get_log_lik_data()`, the `backward_convolutional` branch (line ~737)

This is used when there are **additional observables beyond the selection** (e.g., p), or when `data_lik_from_abundance = False`. It computes the per-cluster conditional pdf $P(\boldsymbol{\omega}_\mathrm{obs} \mid M, z)$ and integrates it against the HMF over $\ln M$.

**Algorithm for each cluster $i$:**

#### Step A: Adaptive mass range estimation (lines ~661–716)

The code first narrows the $\ln M$ integration window around the most likely mass for the cluster:

1. Forward-propagate through all layers of the selection observable to find the mass $M_\mathrm{centre}$ that maps closest to $q_{\mathrm{obs},i}$.
2. At each layer, compute the derivative $dx_1/dx_0$ and the scatter $\sigma_k$.
3. **Backpropagate the scatter** through the layers to estimate the total effective width $\Delta\ln M$ in mass space:
   ```python
   DlnM = 0
   for layer in reverse(layers):
       sigma_k = sqrt(Cov_kk)
       DlnM = sqrt(DlnM^2 + (sigma_k / derivative_k)^2)
   ```
4. Set the integration range to $[\ln M_\mathrm{centre} - n_\sigma \cdot \Delta\ln M,\; \ln M_\mathrm{centre} + n_\sigma \cdot \Delta\ln M]$ where $n_\sigma$ = `sigma_mass_prior` (default 5).
5. Create a uniform grid of `n_points_data_lik` (default 128) points in this range.

#### Step B: Backward convolution (lines ~737–931)

The code then loops over each **correlation set** in `observables_select` (e.g., `["q_so_sim"]` and `["p_so_sim"]` are separate sets).

For each correlation set with $n_\mathrm{obs}$ observables and $n_\mathrm{layer} = 2$ layers:

**Phase 1 — Forward pass through intermediate layers** (layers 0 to $n_\mathrm{layer} - 2$):

```python
# For each intermediate layer, evaluate the scaling relation on the lnM grid:
for layer in layers[0:-1]:         # i.e., layer 0 only when n_layer=2
    x1[j,:] = scaling_relation[obs_j].eval(x[j,:], layer=layer)
    x1_linear[j,:] = linspace(x1[j,0], x1[j,-1], n_points_data_lik)
    x_list.append(x1)
    x_list_linear.append(x1_linear)   # uniformly-spaced grid for convolution
```

**Phase 2 — Backward pass (outermost layer first, then convolve inward):**

Working from the outermost layer backwards:

1. **At the outermost layer** (layer $n_\mathrm{layer} - 1 = 1$):
   
   - Evaluate the scaling relation on the *uniform* intermediate grid `x_p`:
     ```python
     x1[j,:] = scaling_relation[obs_j].eval(x_p[j,:], layer=last) - x_obs[j]
     ```
   - Build an $n_\mathrm{obs}$-dimensional mesh from these residuals:
     ```python
     x_mesh = get_mesh(x1)     # np.meshgrid for n_obs dimensions
     ```
   - Evaluate the multivariate Gaussian pdf with the layer's covariance:
     ```python
     cpdf = eval_gaussian_nd(x_mesh, cov=C[last_layer])
     ```
     This uses the Mahalanobis distance: $\text{pdf} = \frac{1}{\sqrt{(2\pi)^d |\mathbf{C}|}} \exp\!\left(-\tfrac{1}{2}\,\mathbf{x}^T \mathbf{C}^{-1}\mathbf{x}\right)$, computed via direct linear algebra (not `scipy.stats`).

2. **Convolve backwards through intermediate layers** (layer 0):
   
   - Build a Gaussian kernel on a centred grid for the scatter at this layer:
     ```python
     x_p_m[j,:] = x_p[j,:] - mean(x_p[j,:]) + dx/2    # centred grid
     x_p_mesh = get_mesh(x_p_m)
     kernel = eval_gaussian_nd(x_p_mesh, cov=C[layer_0])
     ```
   - **FFT-based N-dimensional convolution** via `scipy.signal.convolve`:
     ```python
     # In utils.py, convolve_nd():
     cpdf = signal.convolve(cpdf, kernel, mode="same") / sum(kernel)
     ```
     This generalises the 1D FFT convolution to $n_\mathrm{obs}$ dimensions (1D for a single observable, 2D for two correlated observables in the same set, etc.).
   
   - **Re-interpolate** the convolved pdf back onto the original (non-uniform) scaling-relation grid using `RegularGridInterpolator` (for $n_\mathrm{obs} > 1$) or `np.interp` (for $n_\mathrm{obs} = 1$).

3. **Extract the diagonal** — for $n_\mathrm{obs} > 1$ within the same correlation set, collapse the mesh by extracting the mass-diagonal (all observables evaluated at the same $\ln M$):
   ```python
   cpdf = extract_diagonal(cpdf)   # shape: (n_lnM,) from (n_lnM, n_lnM, ...)
   ```

**Phase 3 — Multiply correlation sets:**

The conditional pdfs from each correlation set are multiplied together:
```python
cpdf_product = cpdf_product * cpdf    # product over correlation sets
```

When q and p are in **separate** correlation sets (`[["q_so_sim"], ["p_so_sim"]]`), this exploits:
$$P(q, p \mid M, z) = P(q \mid M, z) \times P(p \mid M, z)$$
Each factor is a 1D backward convolution (fast). If they were in the **same** set (`[["q_so_sim", "p_so_sim"]]`), it would be a single 2D backward convolution (slower).

**Phase 4 — Integrate over mass:**

```python
cpdf_with_hmf = cpdf_product * hmf(z) * 4 * pi * f_sky
lik_cluster = simpson(cpdf_with_hmf, x=lnM)
```

The per-cluster log-likelihood is $\ln \mathcal{L}_i = \ln(\text{lik\_cluster})$.

### 6.3 Alternative: direct integral approach

**Entry point:** `cnc.py` → `get_log_lik_data()`, the `direct_integral` branch (line ~948)

When `data_lik_type = "direct_integral"`, the code bypasses backward convolution and evaluates the multi-layer integral by brute force:

1. For each mass grid point $m$:
   - Evaluate the scaling relations at all layers.
   - Evaluate the joint Gaussian pdf at both layer 0 (intrinsic scatter) and layer 1 (measurement noise).
   - Numerically integrate out the intermediate variables using `simpson`.
2. Integrate the result over $\ln M$.

This is exact but much slower ($O(n_\mathrm{lnM} \times n_\mathrm{grid}^{n_\mathrm{obs}})$) and only works with a single correlation set.

### 6.4 Stacked likelihood algorithm

**Entry point:** `cnc.py` → `get_log_lik_stacked()` (line ~1091)

This uses the **mass posterior from the backward convolution** (stored in `cpdf_dict`) as a weight function:

1. For each cluster $i$ in the stacking sample:
   - Retrieve $P(\ln M \mid q_i, z_i)$ from the previously computed `cpdf_dict` (normalised).
   - Evaluate $\langle p \rangle(M, z_i) = e^{\ln\bar{p} + \sigma_{\ln p}^2/2}$ via `scaling_relations.get_mean()`.
   - Compute:
     ```python
     obs_mean_i = simpson(mean_p(lnM) * P(lnM | q_i, z_i), x=lnM)
     obs_var_i  = simpson((var_p(lnM) + mean_p(lnM)^2) * P(lnM), x=lnM) - obs_mean_i^2
     ```

2. Average over all stacked clusters:
   ```python
   stacked_model = sum(obs_mean_i) / N_stack
   stacked_var   = sum(obs_var_i) / N_stack^2
   ```

3. Gaussian log-likelihood:
   ```python
   log_lik = -0.5 * (stacked_obs - stacked_model)^2 / stacked_var
   ```

### 6.5 Parallelisation

**Entry point:** `utils.py` → `launch_multiprocessing()` (line ~232)

cosmocnc uses Python's `multiprocess` (a `multiprocessing` fork) to distribute work across cores:

| Computation | Parallelised over | Parameter |
|---|---|---|
| Abundance grid | Sky patches or redshift bins | `number_cores_abundance` |
| Per-cluster backward convolution | Clusters | `number_cores_data` |
| Stacked likelihood | Clusters | `number_cores_stacked` |

Each worker puts its results into a `multiprocess.Queue`, which the main process collects and merges into `return_dict`.

### 6.6 Key numerical functions in `utils.py`

| Function | What it does | Algorithm |
|----------|-------------|-----------|
| `convolve_1d(x, dn_dx, sigma)` | 1D convolution of a distribution with a Gaussian | `scipy.signal.convolve(dn_dx, kernel, mode="same", method="fft")` — FFT-based by default |
| `convolve_nd(distribution, kernel)` | N-dimensional convolution | `scipy.signal.convolve(distribution, kernel, mode="same")` — uses FFT internally for large arrays |
| `eval_gaussian_nd(x_mesh, cov)` | Evaluate multivariate Gaussian pdf on a mesh | Direct computation via $\mathbf{C}^{-1}$ and Mahalanobis distance; faster than `scipy.stats.multivariate_normal.pdf` |
| `gaussian_1d(x, sigma)` | Evaluate 1D Gaussian | $\frac{1}{\sqrt{2\pi}\sigma}\exp(-x^2/2\sigma^2)$ |
| `get_mesh(x)` | Build N-dimensional meshgrid | `np.meshgrid(x[0], x[1], ...)` |
| `extract_diagonal(tensor)` | Extract mass-diagonal from N-dim array | `np.diag(tensor)` for 2D; explicit loop for 3D |

### 6.7 Summary: which algorithm is used when

| Scenario | Abundance grid | Per-cluster likelihood | Method |
|----------|---------------|----------------------|--------|
| **q only**, `data_lik_from_abundance=True` | Forward FFT convolution through 2 layers | Read from abundance grid (interpolation) | Grid lookup |
| **q + p**, separate correlation sets | Forward FFT convolution (q only, for $N_\mathrm{pred}$) | Backward convolution: two independent 1D FFT convolutions multiplied | `backward_convolutional` |
| **q + p**, same correlation set | Forward FFT convolution (q only, for $N_\mathrm{pred}$) | Backward convolution: one 2D FFT convolution | `backward_convolutional` |
| **q + p**, `data_lik_type="direct_integral"` | Forward FFT convolution (q only, for $N_\mathrm{pred}$) | Brute-force nested numerical integration over intermediate variables | `direct_integral` |
| **Stacked p** | (same as above) | Mass posterior from backward convolution reused as weight | Simpson integration |

---

## 7. Code-to-Maths Mapping

### Key files

| File | Role |
|------|------|
| `surveys/survey_sr_so_sim.py` | Defines scaling relations and scatter for q and p |
| `cosmocnc/cnc.py` | Abundance computation, unbinned/binned likelihood, stacked likelihood |
| `cosmocnc/sr.py` | `covariance_matrix` class (builds the layer-wise covariance) |
| `cosmocnc/params.py` | Default parameter values |

### Key functions in `survey_sr_so_sim.py`

| Function / Method | Mathematical operation |
|------|------|
| `scaling_relations.eval_scaling_relation(x0, layer=0)` for `q_so_sim` | $\ln\bar{q} = \alpha\,x_0 + \mathrm{prefactor} - \ln\sigma_\mathrm{SZ}(\theta_{500})$ |
| `scaling_relations.eval_scaling_relation(x0, layer=1)` for `q_so_sim` | $q = \sqrt{e^{2x_0} + \mathrm{dof}}$ |
| `scaling_relations.eval_scaling_relation(x0, layer=0)` for `p_so_sim` | $\ln\bar{p} = \ln(\mathcal{F}_\mathrm{lens}\,a_\mathrm{lens}\,(0.1\,b_\mathrm{cmblens})^{1/3}) + x_0/3 - \ln\sigma_\mathrm{lens}$ |
| `scaling_relations.eval_scaling_relation(x0, layer=1)` for `p_so_sim` | $p = e^{x_0}$ |
| `scatter.get_cov(layer=0)` | Returns $\sigma_{\ln q}^2$, $\sigma_{\ln p}^2$, or $\rho\,\sigma_{\ln q}\,\sigma_{\ln p}$ |
| `scatter.get_cov(layer=1)` | Returns 1 (diagonal) or 0 (off-diagonal) |
| `scaling_relations.get_mean(x0)` for `p_so_sim` | $\langle p \rangle = e^{\ln\bar{p} + \sigma_{\ln p}^2/2}$ |

### Key parameters

| Parameter | Symbol | Default | Meaning |
|-----------|--------|---------|---------|
| `A_szifi` | $A$ | $-4.3054$ | SZ $Y$-$M$ normalisation (log10) |
| `alpha_szifi` | $\alpha$ | $1.1233$ | SZ $Y$-$M$ slope |
| `bias_sz` | $b_\mathrm{SZ}$ | $0.62$ | Hydrostatic mass bias ($1-b$) for SZ |
| `sigma_lnq_szifi` | $\sigma_{\ln q}$ | $0.173$ | Intrinsic scatter in $\ln q$ |
| `bias_cmblens` | $b_\mathrm{cmblens}$ | $0.92$ | CMB lensing mass bias |
| `a_lens` | $a_\mathrm{lens}$ | $1.0$ | Lensing amplitude calibration |
| `sigma_lnp` | $\sigma_{\ln p}$ | $0.22$ | Intrinsic scatter in $\ln p$ |
| `corr_lnq_lnp` | $\rho_{\ln q,\ln p}$ | $0.77$ | Correlation between $\ln q$ and $\ln p$ intrinsic scatter |
| `dof` | dof | $0$ | Degrees of freedom in $\chi$-like q transformation |
| `q_cutoff` | $q_\mathrm{th}$ | $0$ | Selection threshold (set via `obs_select_min`) |
| `obs_select_min` | — | $6.0$ | Minimum q for catalogue selection |

"""
===========================================================
File        : RFFVolterraSfBM_TestingExamples.py
Project     : LogSfBM_Model
Authors     : Othmane Zarhali, Nicolas Langrené, Jean-François Muzy
Description :
    Runnable examples for RFFVolterraSfBM_Class.py, the OOP wrapper for the
    Random Fourier Features approximation of the SfBM kernel.

    Unlike RFFVolterraSfBM_UnitTesting.py, this file makes NO assertions.
    It runs each class and method against realistic parameters, prints
    actual output (shapes, values, error statistics, ...), and saves
    diagnostic figures to ./figures_sfbmrff/ so you can read it top to
    bottom to understand what each part of the class does and what it
    returns.

    All plots are saved to ./figures_sfbmrff/<name>.png instead of being
    shown on-screen (matplotlib is set to the non-interactive "Agg" backend).

    Examples marked [ORIGINAL] reproduce the concrete calls that appeared
    in the original RFFVolterraSfBM.py script; they are faithfully re-stated
    using the OOP API with the original parameter values quoted explicitly
    for reference.

Contents
--------
    1.  SpectralDensity: numeric vs closed-form comparison (d=1)
    2.  SpectralDensity: grid comparison and error statistics (d=1, d=2)
    3.  SpectralDensity: save comparison to Excel
    4.  SpectralSampler: Hamiltonian MC sampling (d=1)  [ORIGINAL]
    5.  SpectralSampler: Acceptance-Rejection sampling (d=1)
    6.  SpectralSampler: Acceptance-Rejection sampling (d=2)
    7.  SpectralSampler: mc_phi and kernel_rff reconstruction (d=1)
    8.  SpectralSampler: kernel comparison plot and Excel save (d=1) [ORIGINAL]
    9.  SVE_Simulator: single path comparison Euler vs RFF  [ORIGINAL]
    10. SVE_Simulator: elapsed-time benchmark (fast version)
    11. SVE_Simulator: L^p strong error vs M
    12. SVE_Simulator: weak phi-observable error vs M

How to run
----------
    python RFFVolterraSfBM_TestingExamples.py

Figures are saved as PNG in ./figures_sfbmrff/.
All examples are self-contained and do not require any external files.
"""

import os
import math
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── project imports ───────────────────────────────────────────────────────────
from RFFVolterraSfBM_Class import (
    SpectralDensity,
    SpectralSampler,
    SVE_Simulator,
    _generate_grid,
)
from Utils import (
    generate_grid,
    FourierTransformSfBMkernel_numeric,
    FourierTransformSfBMkernel_fromCloseForm,
    mc_phi,
    kernel_reconstruction_RFF,
    lambda_to_latex_mapping,
    EstH_RFF,
    radial_pdf_on_r,
)

# ── output directories ─────────────────────────────────────────────────────────
FIGURES_DIR = "./figures_sfbmrff"
EXCEL_DIR   = "./excel_sfbmrff"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(EXCEL_DIR,   exist_ok=True)

# ── shared parameters (mirror the original script defaults) ────────────────────
NU2     = 50.0       # nu² = 50 → nu = sqrt(50) ≈ 7.07
NU      = math.sqrt(NU2)
H       = 0.1
T       = 200.0
D       = 1

SIGMA   = lambda t, x: 0.3 * (1.0 + 0.1 * x)
X0      = 0.0
T_FINAL = 1.0

print("=" * 70)
print("RFFVolterraSfBM OOP Testing Examples")
print("=" * 70)
print(f"Base parameters: nu²={NU2}, H={H}, T={T}, d={D}")
print()


# ===========================================================================
# 1. SpectralDensity — point evaluation (d=1)
# ===========================================================================
print("─" * 60)
print("Example 1: SpectralDensity point evaluation (d=1)")
print("─" * 60)

sd1 = SpectralDensity(nu2=NU2, H=H, T=T, d=1)

xs_eval = [1.0, 5.0, 10.0, 50.0]
print(f"{'x':>8}  {'f_numeric':>14}  {'f_closedform':>14}  {'abs_err':>12}")
print("-" * 56)
for x_val in xs_eval:
    x_arr  = np.array([x_val])
    f_num  = sd1.spectral_density_numeric(x_arr)
    f_cf   = sd1.spectral_density_closedform(x_arr)
    err    = abs(f_num - f_cf)
    print(f"{x_val:8.1f}  {f_num:14.6e}  {f_cf:14.6e}  {err:12.4e}")
print()


# ===========================================================================
# 2. SpectralDensity — grid comparison and error statistics (d=1)
# ===========================================================================
print("─" * 60)
print("Example 2: SpectralDensity grid comparison (d=1, npts=30)")
print("─" * 60)

pts, Fnum, Frhs, abs_err = sd1.compare(xmin=1.0, xmax=8.0, npts=30)
print(f"Grid size        : {len(pts)} points")
print(f"Max abs error    : {np.max(abs_err):.3e}")
print(f"Mean abs error   : {np.mean(abs_err):.3e}")
print(f"Numeric range    : [{Fnum.min():.4e}, {Fnum.max():.4e}]")
print(f"Closed-form range: [{Frhs.min():.4e}, {Frhs.max():.4e}]")
print()

# Plot
xs = pts[:, 0]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
ax1.plot(xs, Fnum, label="Numeric")
ax1.plot(xs, Frhs, "--", label="Closed form")
ax1.set_ylabel(r"$f_\omega$", fontsize=14)
ax1.set_title(f"Spectral density comparison (d=1, H={H}, T={T})", fontsize=14)
ax1.legend(); ax1.grid(True)
ax2.semilogy(xs, abs_err + 1e-300, label="Absolute error")
ax2.set_xlabel("x", fontsize=14); ax2.set_ylabel("error", fontsize=14)
ax2.grid(True); ax2.legend()
fig.tight_layout()
plt.savefig(f"{FIGURES_DIR}/01_spectraldensity_comparison_d1.png", dpi=150)
plt.close("all")
print(f"Figure saved: 01_spectraldensity_comparison_d1.png")
print()


# ===========================================================================
# 3. SpectralDensity — save comparison to Excel
# ===========================================================================
print("─" * 60)
print("Example 3: SpectralDensity save comparison to Excel")
print("─" * 60)

excel_path = f"{EXCEL_DIR}/spectraldensity_comparison_d1.xlsx"
sd1.save_comparison(excel_path, xmin=1.0, xmax=8.0, npts=20)
print(f"Excel file: {excel_path}")
print()


# ===========================================================================
# 4. SpectralSampler — Hamiltonian MC sampling (d=1)  [ORIGINAL]
# ===========================================================================
print("─" * 60)
print("Example 4: SpectralSampler HMC (d=1, n=8000)  [ORIGINAL]")
print(f"  Original parameters: nu={NU:.3f}, T={T}, H={H}, d=1, n=8000")
print(f"  (Using n=100 here for speed)")
print("─" * 60)

# Original: samp = sample_from_density(8000, nu=nu, T=T, H=H, d=1, method="Hamiltonian MC")
sampler_hmc = SpectralSampler(
    nu2=NU2, H=H, T=T, d=D,
    n_samples=8000,   # 8000 in original; reduced for example speed
    method="Hamiltonian MC",
)
samp_hmc = sampler_hmc.sample()

print(f"Sample shape  : {samp_hmc.shape}")
print(f"Sample range  : [{samp_hmc.min():.4f}, {samp_hmc.max():.4f}]")
print(f"Sample mean   : {samp_hmc.mean():.4f}")
print(f"Sample std    : {samp_hmc.std():.4f}")
print()

# Radial density at a few points (sanity check)
print("Radial density spot-check (p_r):")
for r_val in [0.1, 0.5, 1.0, 2.0]:
    p = radial_pdf_on_r(r_val, nu=NU, T=T, H=H, d=D)
    print(f"  p_r({r_val:.1f}) = {p:.6e}")
print()


# ===========================================================================
# 5. SpectralSampler — mc_phi and kernel_rff reconstruction (d=1)
# ===========================================================================
print("─" * 60)
print("Example 5: mc_phi and kernel_rff reconstruction (d=1)")
print("─" * 60)

u_vals = np.linspace(-T, T, 50)
K_rff  = sampler_hmc.kernel_rff(u_vals.reshape(-1, 1))

# True SfBM kernel
K_true = np.array([
    NU2/2 * (1 - (abs(u)/T)**(2*H)) * (abs(u) <= T)
    for u in u_vals
])
err = np.abs(K_rff - K_true)
print(f"Max abs error (kernel RFF vs true): {err.max():.4e}")
print(f"Mean abs error                    : {err.mean():.4e}")
print()

# mc_phi at u=0 must equal 1
phi0 = sampler_hmc.mc_phi(np.array([0.0]))
print(f"mc_phi(u=0) = {phi0:.10f}  (should be exactly 1.0)")
print()

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
ax1.plot(u_vals, K_rff, label=f"RFF (M={sampler_hmc.n_samples})")
ax1.plot(u_vals, K_true, "--", label="Exact kernel")
ax1.set_ylabel("K(τ)", fontsize=14)
ax1.set_title(f"Kernel reconstruction via RFF (d={D}, H={H}, T={T})", fontsize=14)
ax1.legend(); ax1.grid(True)
ax2.semilogy(u_vals, err + 1e-300, label="abs error")
ax2.set_xlabel("τ", fontsize=14); ax2.set_ylabel("error", fontsize=14)
ax2.grid(True); ax2.legend()
fig.tight_layout()
plt.savefig(f"{FIGURES_DIR}/05_kernel_rff_d1.png", dpi=150)
plt.close("all")
print(f"Figure saved: 05_kernel_rff_d1.png")
print()


# ===========================================================================
# 6. SpectralSampler — Acceptance-Rejection (d=1)
# ===========================================================================
print("─" * 60)
print("Example 6: SpectralSampler Acceptance-Rejection (d=1, n=50)")
print("─" * 60)

sampler_ar1 = SpectralSampler(
    nu2=NU2, H=H, T=T, d=1,
    n_samples=50, method="Acceptance Rejection",
)
samp_ar1 = sampler_ar1.sample()
print(f"Sample shape: {samp_ar1.shape}")
print(f"Sample range: [{samp_ar1.min():.4f}, {samp_ar1.max():.4f}]")
phi_ar = sampler_ar1.mc_phi(np.array([0.0]))
print(f"mc_phi(0)   : {phi_ar:.10f}  (should be 1.0)")
print()


# ===========================================================================
# 7. SpectralSampler — Acceptance-Rejection (d=2)
# ===========================================================================
print("─" * 60)
print("Example 7: SpectralSampler Acceptance-Rejection (d=2, n=50)")
print("─" * 60)

sampler_ar2 = SpectralSampler(
    nu2=NU2, H=H, T=T, d=2,
    n_samples=50, method="Acceptance Rejection",
)
samp_ar2 = sampler_ar2.sample()
print(f"Sample shape     : {samp_ar2.shape}")
print(f"Radii (first 5)  : {np.linalg.norm(samp_ar2[:5], axis=1)}")
phi_ar2 = sampler_ar2.mc_phi(np.array([0.0, 0.0]))
print(f"mc_phi([0,0])    : {phi_ar2:.10f}  (should be 1.0)")
print()


# ===========================================================================
# 8. SpectralSampler — kernel comparison plot and Excel save  [ORIGINAL]
# ===========================================================================
print("─" * 60)
print("Example 8: PlotKernel_RFF equivalent (d=1, t_max=T, n_points=100)")
print("  [ORIGINAL] PlotKernel_RFF(nu2, H, T, samp, d=1, savefile=False)")
print("─" * 60)

excel_kernel = f"{EXCEL_DIR}/kernel_comparison_d1.xlsx"
sampler_hmc.save_kernel_comparison(excel_kernel, t_max=T, n_points=30)
print(f"Excel saved: {excel_kernel}")
sampler_hmc.plot_kernel_comparison(
    t_max=T, n_points=30,
    save_path=f"{FIGURES_DIR}/08_kernel_comparison_d1.png",
)
print(f"Figure saved: 08_kernel_comparison_d1.png")
print()


# ===========================================================================
# 9. SVE_Simulator — single path Euler vs Euler-with-RFF  [ORIGINAL]
# ===========================================================================
print("─" * 60)
print("Example 9: SVE_Simulator single path comparison  [ORIGINAL]")
print(f"  Parameters: H={H}, T={T}, nu={NU:.3f}, N_t=50, M=10")
print("─" * 60)

sim = SVE_Simulator(
    H=H, T=T, nu=NU,
    sigma=SIGMA, x0=X0,
    T_final=T_FINAL, N_t=50,
)

np.random.seed(0)
G = np.random.normal(size=50)

X_euler = sim.simulate("Euler", brownian_path=G)
X_rff   = sim.simulate("Euler with RFF", M=10, brownian_path=G)

print(f"Euler path — shape: {X_euler.shape}, "
      f"min={X_euler.min():.4f}, max={X_euler.max():.4f}")
print(f"RFF path   — shape: {X_rff.shape}, "
      f"min={X_rff.min():.4f}, max={X_rff.max():.4f}")
print(f"Initial value Euler: {X_euler[0]:.6f}  (should be {X0})")
print(f"Initial value RFF  : {X_rff[0]:.6f}    (should be {X0})")

t_grid = np.linspace(0, T_FINAL, 51)
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(t_grid, X_euler, label="Euler")
ax.plot(t_grid, X_rff,   label="Euler with RFF (M=10)", linewidth=1.8)
ax.set_xlabel("t", fontsize=14)
ax.legend(fontsize=12)
fig.tight_layout()
plt.savefig(f"{FIGURES_DIR}/09_path_comparison.png", dpi=150)
plt.close("all")
print(f"Figure saved: 09_path_comparison.png")
print()


# ===========================================================================
# 10. SVE_Simulator — elapsed-time benchmark
# ===========================================================================
print("─" * 60)
print("Example 10: SVE_Simulator elapsed-time benchmark")
print("  N_t_values=[10, 30], M=5, n_repeats=2  (fast version)")
print("─" * 60)

sim_bench = SVE_Simulator(
    H=H, T=T, nu=NU,
    sigma=SIGMA, x0=X0,
    T_final=T_FINAL, N_t=10,
)
df_timing = sim_bench.benchmark_elapsed_time(
    M=5, N_t_values=[10, 30], n_repeats=2
)
print(df_timing.to_string(index=False))
print()


# ===========================================================================
# 11. SVE_Simulator — L^p strong error vs M
# ===========================================================================
print("─" * 60)
print("Example 11: SVE_Simulator L^p strong error vs M")
print("  M_values=[3, 6, 10], p_values=[1, 2], N_MC=5, N_t=20")
print("─" * 60)

sim_err = SVE_Simulator(
    H=H, T=T, nu=NU,
    sigma=SIGMA, x0=X0,
    T_final=T_FINAL, N_t=20,
)
means, stds = sim_err.lp_strong_error(
    M_values=[3, 6, 10],
    p_values=[1, 2],
    N_MC=5,
    same_brownian=False,
    save_path=f"{EXCEL_DIR}/lp_strong_error.xlsx",
)
for p in [1, 2]:
    print(f"  p={p}: means = {[f'{v:.4e}' for v in means[p]]}")
print(f"Excel + figure saved in {EXCEL_DIR}/")
print()


# ===========================================================================
# 12. SVE_Simulator — weak phi-observable error vs M
# ===========================================================================
print("─" * 60)
print("Example 12: SVE_Simulator weak phi-observable error vs M")
print("  phi_functions=[x, x²], M_values=[3, 6, 10], N_MC=5, N_t=20")
print("─" * 60)

phi_list = [lambda x: x, lambda x: x ** 2]
labels   = [lambda_to_latex_mapping(phi) for phi in phi_list]
print(f"  phi labels: {labels}")

err_arr = sim_err.weak_phi_error(
    M_values=[3, 6, 10],
    phi_functions=phi_list,
    N_MC=5,
    same_brownian=True,
    error_type="weak",
    save_path=f"{EXCEL_DIR}/weak_phi_error.xlsx",
)
for i, label in enumerate(labels):
    print(f"  {label}: errors = {[f'{v:.4e}' for v in err_arr[i]]}")
print(f"Excel + figure saved in {EXCEL_DIR}/")
print()


# ===========================================================================
# 13. Utils standalone functions — spot checks
# ===========================================================================
print("─" * 60)
print("Example 13: Utils standalone function spot-checks")
print("─" * 60)

# FourierTransformSfBMkernel_numeric vs closedform
x_test = np.array([2.0])
f_num  = FourierTransformSfBMkernel_numeric(x_test, nu2=NU2, H=H, T=T, d=1)
f_cf   = FourierTransformSfBMkernel_fromCloseForm(x_test, nu2=NU2, H=H, T=T, d=1)
print(f"FT numeric   at x=2 : {f_num:.6e}")
print(f"FT closedform at x=2: {f_cf:.6e}")
print(f"Relative error      : {abs(f_num - f_cf)/max(abs(f_num),abs(f_cf),1e-15):.3e}")
print()

# EstH_RFF
np.random.seed(1)
xvals = np.random.normal(size=(1, 500)) * 0.05
H_est, l2_est = EstH_RFF(xvals)
print(f"EstH_RFF on synthetic path: H_est={H_est:.4f}, lambda2_est={l2_est:.6f}")
print()

# mc_phi from standalone function
rng_samp  = np.random.default_rng(42)
samp_util = rng_samp.normal(size=(100, 1))
phi_val   = mc_phi(samp_util, np.array([1.0]))
print(f"mc_phi (standalone, u=1.0): {phi_val:.6f}")
k_vals = kernel_reconstruction_RFF(samp_util, np.array([[0.0]]), nu2=NU2)
print(f"kernel_rff at origin       : {k_vals[0]:.6f}  (expected {NU2/2:.6f})")
print()


# ===========================================================================
# Summary
# ===========================================================================
print("=" * 70)
print("All examples completed successfully.")
print(f"Figures saved in : {FIGURES_DIR}/")
print(f"Excel files in   : {EXCEL_DIR}/")
print("=" * 70)

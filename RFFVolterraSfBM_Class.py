"""
===========================================================
File        : RFFVolterraSfBM_Class.py
Project     : LogSfBM_Model
Authors     : Othmane Zarhali, Nicolas Langrené, Jean-François Muzy
Description :
    Object-oriented implementation of the Random Fourier Features (RFF)
    approximation framework for the Stationary fractional Brownian Motion
    (SfBM) kernel. The module provides three collaborating classes:

    SpectralDensity
        Computes the spectral density f_omega(x) of the SfBM kernel via
        a Fourier-transform identity involving hypergeometric functions
        (closed-form) or a direct numerical radial integral (numeric),
        and supports comparison and Excel-saving of both.

    SpectralSampler
        Draws samples from the spectral density f_omega using either
        Hamiltonian Monte Carlo (HMC) or Acceptance-Rejection (AR),
        and exposes the cosine characteristic function estimator
        E[cos(eta^T u)] that reconstructs the kernel via RFF.

    SVE_Simulator
        Simulates a Stochastic Volterra Equation (SVE) with the SfBM
        kernel using either the brute-force Euler scheme (O(N^2)) or
        the fast Euler-with-RFF scheme (O(N·M)), and provides routines
        for elapsed-time benchmarking and L^p / weak-error analysis.

Architecture
------------
SpectralDensity(nu2, H, T, d)
    spectral_density_numeric(x)         -- numerical FT at point x
    spectral_density_closedform(x)      -- hypergeometric closed form at x
    compare(xmin, xmax, npts)           -- grid comparison + error stats
    save_comparison(path, xmin, xmax, npts) -- save results to Excel

SpectralSampler(nu2, H, T, d, n_samples, method)
    sample()                            -- draw n_samples from f_omega
    kernel_rff(u_list)                  -- RFF kernel reconstruction K^M(u)
    mc_phi(u, return_samples)           -- E[cos(eta^T u)] estimator

SVE_Simulator(H, T, nu, sigma, x0, T_final, N_t)
    simulate(algo, M, brownian_path)    -- single path: 'Euler' or 'Euler with RFF'
    compare_paths(M)                    -- plot Euler vs RFF side by side
    benchmark_elapsed_time(M_values, n_repeats) -- timing study
    lp_strong_error(M_values, p_values, N_MC)   -- strong L^p error vs M
    weak_phi_error(M_values, phi_list, N_MC)    -- weak observable error vs M

Original standalone functions preserved as static methods on each class
for backward-compatibility with existing call sites.
"""

from __future__ import annotations

import os
import ast
import math
import time
import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import mpmath as mp
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import integrate, special
from scipy.integrate import quad
try:
    from scipy.integrate import simps
except ImportError:
    from scipy.integrate import simpson as simps
from scipy.special import jv, gamma as sp_gamma
from scipy.stats import f as f_dist, uniform

mp.mp.dps = 50  # default high-precision setting


# ---------------------------------------------------------------------------
# Helper: generate a d-dimensional grid with 0 included
# ---------------------------------------------------------------------------

def _generate_grid(
    d: int = 1,
    xmin: float = 1.0,
    xmax: float = 5.0,
    npts: int = 20,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Generate a d-dimensional evaluation grid in [xmin, xmax]^d,
    ensuring 0.0 is always included in each axis.

    Returns
    -------
    pts  : ndarray, shape (npts^d, d)
    axes : list of d 1-D arrays
    """
    axes = [
        np.unique(np.concatenate((np.linspace(xmin, xmax, npts), [0.0])))
        for _ in range(d)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    pts  = np.stack([m.reshape(-1) for m in mesh], axis=-1)
    return pts, axes


# ===========================================================================
# Class 1: SpectralDensity
# ===========================================================================

class SpectralDensity:
    """Spectral density of the SfBM kernel and its Fourier comparison tools.

    The SfBM kernel is
        K(tau) = (nu²/2) * (1 - (||tau||/T)^{2H}) * 1_{||tau|| <= T}

    Its Fourier transform (spectral density) has the closed-form:
        f_omega(x) = pref * (term0 - term1)
    where
        pref  = (T²/2)^{d/2} * nu / Gamma(d/2)
        term0 = (1/d)     * 0F1(; d/2+1; -T²||x||²/4)
        term1 = (1/(d+2H))* 1F2(d/2+H; d/2, d/2+H+1; -T²||x||²/4)

    Parameters
    ----------
    nu2 : float
        Squared amplitude parameter nu².
    H   : float
        Hurst exponent in (0, 1/2).
    T   : float
        Integral scale.
    d   : int
        Dimension.
    """

    def __init__(self, nu2: float, H: float, T: float, d: int = 1) -> None:
        self.nu2 = float(nu2)
        self.H   = float(H)
        self.T   = float(T)
        self.d   = int(d)

    # ------------------------------------------------------------------
    # Numeric spectral density (radial integral)
    # ------------------------------------------------------------------

    def spectral_density_numeric(
        self,
        x: Union[float, np.ndarray],
        method: str = "simpson",
    ) -> float:
        """Numerically compute f_omega(x) via a radial Hankel integral.

        For d=1 uses scipy.integrate.quad with cosine weight.
        For d>=2 uses adaptive Simpson on the Bessel integrand.

        Parameters
        ----------
        x      : scalar or array representing the evaluation point.
        method : 'simpson' (default) for d>=2.

        Returns
        -------
        float
        """
        nu2, H, T, d = self.nu2, self.H, self.T, self.d
        x = np.asarray(x, dtype=float)

        if d == 1:
            scalar_x = float(x.ravel()[0]) if x.size > 0 else float(x)
            f = lambda u: nu2 / 2.0 * (1.0 - (abs(u) / T) ** (2.0 * H))
            val, _ = integrate.quad(
                f, 0, T, weight="cos", wvar=scalar_x,
                epsabs=1e-12, epsrel=1e-12,
            )
            return 2.0 * val / (2.0 * math.pi) ** 0.5

        # d >= 2: radial Bessel integration
        k = float(np.linalg.norm(x))
        if k == 0.0:
            if d == 2:
                return (nu2 / 2.0) * (H / (2.0 * (1.0 + H))) * T ** 2
            return 0.0

        nu_b = d / 2.0 - 1.0
        U    = k * T
        points_per_wavelength = 20
        n_osc  = max(1.0, U / (2.0 * math.pi))
        N_base = int(math.ceil(n_osc * points_per_wavelength))
        if N_base % 2 == 1:
            N_base += 1

        prev_val = None
        tol_rel  = 1e-6
        tol_abs  = 1e-12
        I        = 0.0
        for mult in [1, 2, 4, 8, 16, 32]:
            N   = N_base * mult
            u   = np.linspace(0.0, U, N + 1)
            hu  = (1.0 - (u / U) ** (2.0 * H)) * u ** (d / 2.0) * jv(nu_b, u)
            Iu  = simps(hu, u)
            I   = (nu2 / 2.0) * Iu / (k ** (d / 2.0 + 1.0))
            if prev_val is not None:
                err = abs(I - prev_val)
                rel = err / max(abs(I), tol_abs, abs(prev_val))
                if rel < tol_rel or err < tol_abs:
                    break
            prev_val = I
        return float(I)

    # ------------------------------------------------------------------
    # Closed-form spectral density (hypergeometric series)
    # ------------------------------------------------------------------

    def spectral_density_closedform(
        self,
        x: Union[float, np.ndarray],
    ) -> float:
        """Evaluate f_omega(x) via the hypergeometric closed-form expression.

        Uses mpmath for high-precision 0F1 and 1F2 evaluation.

        Parameters
        ----------
        x : scalar or d-dimensional array.

        Returns
        -------
        float
        """
        nu2, H, T, d = self.nu2, self.H, self.T, self.d
        x = np.asarray(x, dtype=float)
        r = float(np.linalg.norm(x))
        nu = mp.sqrt(nu2)
        z  = -(T ** 2 * r ** 2) / 4.0

        # Use the factored hypergeometric representation
        norm_x_sq = r ** 2
        pref = (
            nu ** 2 * T ** d
            / (
                2 / (math.pi ** (d / 2))
                * (d / (2 * H) + 1)
                * sp_gamma(d / 2 + 1)
            )
        )
        z_hp = -(norm_x_sq * T ** 2) / 4.0
        hypergeom = mp.hyper(
            [d / 2 + H],
            [d / 2 + H + 1, d / 2 + 1],
            z_hp,
        )
        return float((pref * hypergeom).real)

    # ------------------------------------------------------------------
    # Grid comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        xmin: float = 1.0,
        xmax: float = 8.0,
        npts: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate both numeric and closed-form spectral densities on a grid.

        Parameters
        ----------
        xmin, xmax : float
            Grid boundaries (same for each dimension).
        npts : int
            Number of grid points per dimension.

        Returns
        -------
        pts     : ndarray, shape (N_pts, d)
        Fnum    : ndarray, shape (N_pts,)   numeric values
        Frhs    : ndarray, shape (N_pts,)   closed-form values
        abs_err : ndarray, shape (N_pts,)   absolute error
        """
        pts, _ = _generate_grid(self.d, xmin, xmax, npts)
        Fnum = np.zeros(len(pts))
        Frhs = np.zeros(len(pts))
        for i, xv in enumerate(pts):
            Fnum[i] = self.spectral_density_numeric(xv)
            Frhs[i] = self.spectral_density_closedform(xv)
        abs_err = np.abs(Fnum - Frhs)
        print(f"Max abs error: {np.max(abs_err):.3e}")
        return pts, Fnum, Frhs, abs_err

    # ------------------------------------------------------------------
    # Save to Excel
    # ------------------------------------------------------------------

    def save_comparison(
        self,
        path: str,
        xmin: float = 1.0,
        xmax: float = 8.0,
        npts: int = 30,
    ) -> str:
        """Evaluate and save spectral-density comparison to Excel.

        Parameters
        ----------
        path : str
            Output .xlsx file path.

        Returns
        -------
        str : path to the written file.
        """
        pts, Fnum, Frhs, abs_err = self.compare(xmin, xmax, npts)
        rel_err = abs_err / np.maximum(np.maximum(np.abs(Fnum), np.abs(Frhs)), 1e-15)
        df = pd.DataFrame({
            "pt"     : [tuple(p) for p in pts],
            "Fnum"   : Fnum,
            "Frhs"   : Frhs,
            "abs_err": abs_err,
            "rel_err": rel_err,
        })
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        df.to_excel(path, index=False)
        print(f"Saved: {path}")
        return path

    # ------------------------------------------------------------------
    # Inverse Fourier transform (kernel reconstruction check)
    # ------------------------------------------------------------------

    def inverse_fourier_transform(
        self,
        x: Union[float, np.ndarray],
    ) -> float:
        """Compute the inverse Fourier transform of f_omega at point x,
        which should recover the SfBM kernel K(x).

        Uses the radial Hankel representation for d >= 2.

        Parameters
        ----------
        x : evaluation point.

        Returns
        -------
        float
        """
        nu2, H, T, d = self.nu2, self.H, self.T, self.d
        x = np.asarray(x, dtype=float)

        if d == 1:
            scalar_x = float(x.ravel()[0])
            integrand_real = lambda u: float(
                self.spectral_density_closedform(np.array([u]))
            ) * math.cos(u * scalar_x)
            real_part, _ = quad(integrand_real, -100 * T, 100 * T, limit=200)
            return real_part / (2.0 * math.pi) ** 0.5

        rho = float(np.linalg.norm(x))
        order = d / 2.0 - 1.0

        def integrand_r(r):
            xvec = np.array([r] + [0.0] * (d - 1))
            return (r ** (d / 2.0)) * self.spectral_density_closedform(xvec) * special.jv(order, rho * r)

        I, _ = integrate.quad(integrand_r, 0, 100 * T, epsabs=1e-9, epsrel=1e-8, limit=200)
        return float(rho ** (-(d / 2.0 - 1.0)) * I)


# ===========================================================================
# Class 2: SpectralSampler
# ===========================================================================

class SpectralSampler:
    """Draw samples from the SfBM spectral density and reconstruct the kernel.

    Two sampling methods are available:

    'Hamiltonian MC' (HMC)
        Gradient-based Markov chain sampler; no proposal tuning needed
        and works well in moderate dimensions.

    'Acceptance Rejection' (AR)
        Uses an F-distribution proposal for the radial component,
        combined with uniform directions.  Efficient in d=1 or d=2.

    Parameters
    ----------
    nu2      : float
    H        : float  Hurst exponent
    T        : float  integral scale
    d        : int    dimension (1 or 2 for AR; any for HMC)
    n_samples: int    number of spectral samples eta to draw
    method   : str    'Hamiltonian MC' or 'Acceptance Rejection'
    """

    def __init__(
        self,
        nu2: float,
        H: float,
        T: float,
        d: int = 1,
        n_samples: int = 1000,
        method: str = "Hamiltonian MC",
    ) -> None:
        self.nu2      = float(nu2)
        self.H        = float(H)
        self.T        = float(T)
        self.d        = int(d)
        self.n_samples = int(n_samples)
        self.method   = method
        self._samples: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Radial density (unnormalized)
    # ------------------------------------------------------------------

    def _radial_pdf(self, r: float) -> float:
        """Radial spectral density p_r(r) ∝ f_omega(r·e_1) / (nu²/2)."""
        nu2, H, T, d = self.nu2, self.H, self.T, self.d
        nu = mp.sqrt(nu2)
        z  = -(T ** 2 * r ** 2) / 4.0
        pref = (T ** 2 / 2) ** (d / 2) * nu ** 2 / mp.gamma(d / 2)
        term1 = (1 / d) * mp.hyper([], [d / 2 + 1], z)
        term2 = (1 / (d + 2 * H)) * mp.hyper([d / 2 + H], [d / 2, d / 2 + H + 1], z)
        val = float((pref * (term1 - term2)).real)
        return max(val, 0.0) / (nu2 / 2.0)

    # ------------------------------------------------------------------
    # Sampling backends
    # ------------------------------------------------------------------

    def _sample_hmc(self) -> np.ndarray:
        """Hamiltonian Monte Carlo sampler for the spectral density."""
        n, d = self.n_samples, self.d
        step_size  = 0.05
        n_leapfrog = 20

        def numerical_grad(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
            grad = np.zeros_like(x)
            log_f = lambda v: math.log(max(self._radial_pdf(float(np.linalg.norm(v))), 1e-300))
            for i in range(x.size):
                xp, xm = x.copy(), x.copy()
                xp[i] += eps; xm[i] -= eps
                grad[i] = (log_f(xp) - log_f(xm)) / (2.0 * eps)
            return grad

        samples = np.zeros((n, d))
        x = np.zeros(d)
        logpx = math.log(max(self._radial_pdf(float(np.linalg.norm(x))), 1e-300))

        for i in range(n):
            p    = np.random.normal(size=d)
            p0   = p.copy()
            grad = numerical_grad(x)
            p    = p + 0.5 * step_size * grad
            x_p  = x.copy()
            for _ in range(n_leapfrog):
                x_p  = x_p + step_size * p
                g_p  = numerical_grad(x_p)
                if _ != n_leapfrog - 1:
                    p = p + step_size * g_p
            p = p + 0.5 * step_size * g_p

            logp_p   = math.log(max(self._radial_pdf(float(np.linalg.norm(x_p))), 1e-300))
            H_cur    = -logpx  + 0.5 * np.dot(p0, p0)
            H_prop   = -logp_p + 0.5 * np.dot(p, p)
            if math.log(max(np.random.rand(), 1e-300)) < -(H_prop - H_cur):
                x     = x_p
                logpx = logp_p
            samples[i] = x
        return samples

    def _sample_ar(self) -> np.ndarray:
        """Acceptance-Rejection sampler using an F-distribution proposal."""
        d = self.d
        if d not in (1, 2):
            raise ValueError("Acceptance-Rejection sampler supports only d=1 or d=2.")

        p_dfn, p_dfd, p_scale = 2.25, 0.25, 100.0

        def proposal(r: float) -> float:
            return 0.45 * p_scale * f_dist.pdf(r, dfn=p_dfn, dfd=p_dfd, scale=p_scale)

        n = self.n_samples
        R_eta = np.zeros(n)
        m = 0
        while m < n:
            R_f = f_dist.rvs(dfn=p_dfn, dfd=p_dfd, scale=p_scale)
            num = abs(self._radial_pdf(R_f))
            den = proposal(R_f)
            if den > 0 and uniform.rvs() < num / den:
                R_eta[m] = R_f
                m += 1

        if d == 1:
            return R_eta.reshape(-1, 1)

        # d == 2: random directions
        Z   = np.random.normal(size=(n, 2))
        nrm = np.linalg.norm(Z, axis=1, keepdims=True)
        return Z / nrm * R_eta.reshape(-1, 1)

    # ------------------------------------------------------------------
    # Public: draw samples
    # ------------------------------------------------------------------

    def sample(self) -> np.ndarray:
        """Draw ``n_samples`` from the spectral density f_omega.

        Returns
        -------
        np.ndarray, shape (n_samples, d)
        """
        if self.method == "Hamiltonian MC":
            self._samples = self._sample_hmc()
        elif self.method == "Acceptance Rejection":
            self._samples = self._sample_ar()
        else:
            raise ValueError(f"Unknown method: {self.method!r}. "
                             "Choose 'Hamiltonian MC' or 'Acceptance Rejection'.")
        return self._samples

    def get_samples(self) -> np.ndarray:
        """Return cached samples (or draw them if not yet done)."""
        if self._samples is None:
            self.sample()
        return self._samples

    # ------------------------------------------------------------------
    # Public: characteristic-function estimator and kernel reconstruction
    # ------------------------------------------------------------------

    def mc_phi(
        self,
        u: Union[np.ndarray, float],
        return_samples: bool = False,
    ) -> Union[float, np.ndarray]:
        """Monte-Carlo estimate of E[cos(eta^T u)].

        Parameters
        ----------
        u             : evaluation point, shape (d,) or scalar for d=1.
        return_samples: if True return the per-sample cosine values (shape (1,n)).

        Returns
        -------
        float or ndarray
        """
        eta = self.get_samples()        # (n, d)
        u   = np.atleast_2d(u)          # (1, d)
        dots    = eta @ u.T             # (n, 1)
        cosvals = np.cos(dots)          # (n, 1)
        if return_samples:
            return cosvals.reshape(1, -1)
        return float(cosvals.mean())

    def kernel_rff(
        self,
        u_list: Union[np.ndarray, List],
    ) -> np.ndarray:
        """Reconstruct the SfBM kernel K^M(u) = (nu²/2) * E[cos(eta^T u)].

        Parameters
        ----------
        u_list : array of evaluation points, shape (n_pts, d) or (n_pts,).

        Returns
        -------
        np.ndarray, shape (n_pts,)
        """
        u_list = np.atleast_2d(u_list)
        K0     = self.nu2 / 2.0
        return np.array([K0 * self.mc_phi(u) for u in u_list])

    # ------------------------------------------------------------------
    # Plot: kernel reconstruction comparison
    # ------------------------------------------------------------------

    def plot_kernel_comparison(
        self,
        t_max: Optional[float] = None,
        n_points: int = 100,
        direction: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot RFF kernel reconstruction vs exact SfBM kernel.

        For d=1: plots K^M(u) and K(u) vs u in [-t_max, t_max].
        For d=2: 2-D colormap (using direction slice for d>2).

        Parameters
        ----------
        t_max     : float, plot range (defaults to self.T).
        n_points  : int, number of evaluation points per axis.
        direction : ndarray, direction for d>2 slices.
        save_path : str, if given save PDF there.
        """
        nu2, H, T, d = self.nu2, self.H, self.T, self.d
        t_max = t_max or T
        K_true = lambda x: nu2/2*(1-(np.linalg.norm(x)/T)**(2*H))*(np.linalg.norm(x) <= T)

        pts, axes = _generate_grid(d, -t_max, t_max, n_points)
        K_rff  = np.array([nu2/2 * self.mc_phi(p) for p in pts])
        K_cf   = np.array([K_true(p) for p in pts])
        err    = np.abs(K_rff - K_cf)

        if d == 1:
            xs = axes[0]
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
            ax1.plot(xs, K_rff, label=f"RFF (M={self.n_samples})")
            ax1.plot(xs, K_cf, "--", label="Exact")
            ax1.legend(); ax1.set_ylabel("K(τ)"); ax1.grid(True)
            ax1.set_title(f"Kernel comparison d={d}, H={H}, T={T}")
            ax2.semilogy(xs, err + 1e-300, label="abs error")
            ax2.set_xlabel("τ"); ax2.set_ylabel("error"); ax2.grid(True)
            fig.tight_layout()
        elif d == 2:
            xs = sorted(set(p[0] for p in pts))
            ys = sorted(set(p[1] for p in pts))
            nx, ny = len(xs), len(ys)
            from matplotlib import cm
            fig = plt.figure(figsize=(18, 5))
            for idx, (data, title) in enumerate([(K_rff, "RFF"), (K_cf, "Exact"), (err, "Error")]):
                ax = fig.add_subplot(1, 3, idx+1, projection="3d")
                X, Y = np.meshgrid(xs, ys, indexing="ij")
                ax.plot_surface(X, Y, data.reshape(nx, ny), cmap=cm.cividis, linewidth=0)
                ax.set_title(title)
            fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
        plt.close("all")

    # ------------------------------------------------------------------
    # Save kernel comparison to Excel
    # ------------------------------------------------------------------

    def save_kernel_comparison(
        self,
        path: str,
        t_max: Optional[float] = None,
        n_points: int = 100,
    ) -> str:
        """Evaluate and save RFF vs exact kernel to Excel.

        Returns path of the written file.
        """
        nu2, H, T, d = self.nu2, self.H, self.T, self.d
        t_max  = t_max or T
        K_true = lambda x: nu2/2*(1-(np.linalg.norm(x)/T)**(2*H))*(np.linalg.norm(x) <= T)
        pts, _ = _generate_grid(d, -t_max, t_max, n_points)

        K_rff = np.array([nu2/2 * self.mc_phi(p) for p in pts])
        K_cf  = np.array([K_true(p) for p in pts])
        err   = np.abs(K_rff - K_cf)

        df = pd.DataFrame({
            "pt"      : [tuple(p) for p in pts],
            "RFF"     : K_rff,
            "ClosedForm": K_cf,
            "abs_err" : err,
            "rel_err" : err,
        })
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        df.to_excel(path, index=False)
        print(f"Saved: {path}")
        return path


# ===========================================================================
# Class 3: SVE_Simulator
# ===========================================================================

class SVE_Simulator:
    """Simulate a Stochastic Volterra Equation (SVE) with the SfBM kernel.

    The SVE reads:
        X_t = x0 + ∫_0^t K(t-s) sigma(s, X_s) dW_s

    Two algorithms are available:

    'Euler'
        Brute-force O(N²) Euler–Maruyama scheme with the exact SfBM kernel.

    'Euler with RFF'
        O(N·M) scheme that replaces K by its RFF approximation
            K^M(t) = (K(0)/M) Σ_{m=1}^M cos(η_m t)
        and accumulates the stochastic integrals recursively.

    Parameters
    ----------
    H, T, nu : float  SfBM model parameters (nu = sqrt(nu²))
    sigma    : callable(t, x) -> float  volatility function
    x0       : float  initial condition
    T_final  : float  simulation horizon
    N_t      : int    number of time steps
    """

    def __init__(
        self,
        H: float,
        T: float,
        nu: float,
        sigma: Callable[[float, float], float],
        x0: float = 0.0,
        T_final: float = 1.0,
        N_t: int = 1000,
    ) -> None:
        self.H       = float(H)
        self.T       = float(T)
        self.nu      = float(nu)
        self.nu2     = float(nu ** 2)
        self.sigma   = sigma
        self.x0      = float(x0)
        self.T_final = float(T_final)
        self.N_t     = int(N_t)
        self._K      = lambda t: (self.nu2 / 2.0) * (1.0 - (abs(t) / T) ** (2.0 * H)) * (abs(t) <= T)

    # ------------------------------------------------------------------
    # Internal: simulate one path
    # ------------------------------------------------------------------

    def simulate(
        self,
        algo: str = "Euler with RFF",
        M: int = 100,
        brownian_path: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Simulate one path of the SVE.

        Parameters
        ----------
        algo          : 'Euler' or 'Euler with RFF'.
        M             : number of RFF features (only for 'Euler with RFF').
        brownian_path : optional pre-drawn increments dW of length N_t.

        Returns
        -------
        X : ndarray, shape (N_t + 1,)
        """
        t_grid = np.linspace(0, self.T_final, self.N_t + 1)
        N      = self.N_t
        G      = brownian_path if brownian_path is not None else np.random.normal(size=N)

        if algo == "Euler":
            return self._euler_slow(t_grid, G)
        elif algo == "Euler with RFF":
            return self._euler_rff(t_grid, G, M)
        else:
            raise ValueError(f"Unknown algo {algo!r}. Choose 'Euler' or 'Euler with RFF'.")

    def _euler_slow(self, t_grid: np.ndarray, G: np.ndarray) -> np.ndarray:
        """Brute-force O(N²) Euler–Maruyama scheme."""
        N  = len(t_grid) - 1
        X  = np.zeros(N + 1)
        X[0] = self.x0
        for n in range(N):
            val = self.x0
            for i in range(n):
                dt = t_grid[i + 1] - t_grid[i]
                val += self._K(t_grid[n] - t_grid[i]) * self.sigma(t_grid[i], X[i]) * math.sqrt(dt) * G[i]
            X[n + 1] = val
        return X

    def _euler_rff(
        self,
        t_grid: np.ndarray,
        G: np.ndarray,
        M: int,
    ) -> np.ndarray:
        """O(N·M) Euler scheme with RFF kernel approximation."""
        N    = len(t_grid) - 1
        # draw spectral frequencies
        sampler = SpectralSampler(
            nu2=self.nu2, H=self.H, T=self.T,
            d=1, n_samples=M, method="Hamiltonian MC",
        )
        eta = sampler.sample().ravel()   # (M,)

        X     = np.zeros(N + 1)
        X[0]  = self.x0
        I_c   = np.zeros(M)
        I_s   = np.zeros(M)
        K0    = self._K(0.0)

        cos_eta_t = np.cos(np.outer(eta, t_grid))  # (M, N+1)
        sin_eta_t = np.sin(np.outer(eta, t_grid))

        for n in range(N):
            dt        = t_grid[n + 1] - t_grid[n]
            sig_val   = self.sigma(t_grid[n], X[n])
            I_c      += cos_eta_t[:, n] * sig_val * math.sqrt(dt) * G[n]
            I_s      += sin_eta_t[:, n] * sig_val * math.sqrt(dt) * G[n]
            X[n + 1]  = self.x0 + (K0 / M) * np.sum(
                cos_eta_t[:, n + 1] * I_c + sin_eta_t[:, n + 1] * I_s
            )
        return X

    # ------------------------------------------------------------------
    # Compare paths side by side
    # ------------------------------------------------------------------

    def compare_paths(
        self,
        M: int = 100,
        same_brownian: bool = True,
        save_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate and plot one path for each algorithm.

        Parameters
        ----------
        M             : RFF feature count.
        same_brownian : share the same Brownian increments.
        save_path     : optional PDF output path.

        Returns
        -------
        (X_euler, X_rff)
        """
        t_grid = np.linspace(0, self.T_final, self.N_t + 1)
        G      = np.random.normal(size=self.N_t)
        bw     = G if same_brownian else None

        X_euler = self.simulate("Euler", M=M, brownian_path=G)
        X_rff   = self.simulate("Euler with RFF", M=M, brownian_path=bw)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(t_grid, X_euler, label="Euler")
        ax.plot(t_grid, X_rff,   label=f"Euler with RFF (M={M})", linewidth=1.8)
        ax.set_xlabel("t")
        ax.legend()
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
        plt.close("all")
        return X_euler, X_rff

    # ------------------------------------------------------------------
    # Elapsed-time benchmark
    # ------------------------------------------------------------------

    def benchmark_elapsed_time(
        self,
        M: int,
        N_t_values: List[int],
        n_repeats: int = 1,
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Measure elapsed time of Euler vs Euler-with-RFF for multiple N_t.

        Parameters
        ----------
        M          : RFF feature count.
        N_t_values : list of N_t to test.
        n_repeats  : averages over this many independent runs per N_t.
        save_path  : if given, save results to Excel there.

        Returns
        -------
        pd.DataFrame with columns N_t, Elapsed_time_Euler, Elapsed_time_RFF.
        """
        times_euler = []
        times_rff   = []

        for N_t in N_t_values:
            sim_copy = SVE_Simulator(
                H=self.H, T=self.T, nu=self.nu,
                sigma=self.sigma, x0=self.x0,
                T_final=self.T_final, N_t=N_t,
            )
            t_eu_list  = []
            t_rff_list = []

            for _ in range(n_repeats):
                t0 = time.time()
                sim_copy.simulate("Euler")
                t_eu_list.append(time.time() - t0)

                t0 = time.time()
                sim_copy.simulate("Euler with RFF", M=M)
                t_rff_list.append(time.time() - t0)

            times_euler.append(float(np.mean(t_eu_list)))
            times_rff.append(float(np.mean(t_rff_list)))
            print(f"N_t={N_t}: Euler {times_euler[-1]:.3f}s, RFF {times_rff[-1]:.3f}s")

        df = pd.DataFrame({
            "N_t"                 : N_t_values,
            "Elapsed_time_Euler"  : times_euler,
            "Elapsed_time_RFF"    : times_rff,
        })

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            df.to_excel(save_path, index=False)
            print(f"Saved: {save_path}")
        return df

    # ------------------------------------------------------------------
    # Strong L^p error vs M
    # ------------------------------------------------------------------

    def lp_strong_error(
        self,
        M_values: List[int],
        p_values: List[float],
        N_MC: int = 30,
        same_brownian: bool = False,
        save_path: Optional[str] = None,
    ) -> Tuple[dict, dict]:
        """Monte-Carlo strong L^p error between Euler and Euler-with-RFF.

        Computes E[sup_i |X(t_i) - X^M(t_i)|^p] for each M and p.

        Parameters
        ----------
        M_values      : list of RFF feature counts.
        p_values      : list of p values.
        N_MC          : number of Monte-Carlo paths.
        same_brownian : share Brownian path between schemes.
        save_path     : optional Excel output.

        Returns
        -------
        (results_mean, results_std) : dicts keyed by p.
        """
        results_mean = {p: [] for p in p_values}
        results_std  = {p: [] for p in p_values}

        for M in M_values:
            lp_vals = {p: [] for p in p_values}
            for _ in range(N_MC):
                G   = np.random.normal(size=self.N_t)
                bw  = G if same_brownian else None
                X_e = self.simulate("Euler", M=M, brownian_path=G)
                X_r = self.simulate("Euler with RFF", M=M, brownian_path=bw)
                diff = np.abs(X_e - X_r)
                for p in p_values:
                    lp_vals[p].append(np.max(diff) ** p)

            for p in p_values:
                results_mean[p].append(float(np.mean(lp_vals[p])))
                results_std[p].append(float(np.std(lp_vals[p])))
            print(f"M={M} done.")

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        for p in p_values:
            ax.errorbar(
                M_values, results_mean[p], yerr=results_std[p],
                marker="o", capsize=4,
                label=(f"p = {p}" if p != np.inf else r"p = $\infty$"),
            )
        ax.set_xlabel("M")
        ax.set_ylabel(r"$\|\sup_i |X_i - X_i^M|\|^p_{\mathcal{L}^p}$")
        ax.legend(); ax.grid(True)
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path.replace(".xlsx", ".pdf"), bbox_inches="tight")
            rows = []
            for p in p_values:
                for i, M in enumerate(M_values):
                    rows.append({"M": M, "p": p,
                                 "mean": results_mean[p][i],
                                 "std": results_std[p][i]})
            pd.DataFrame(rows).to_excel(save_path, index=False)
            print(f"Saved: {save_path}")
        plt.close("all")
        return results_mean, results_std

    # ------------------------------------------------------------------
    # Weak phi-observable error vs M
    # ------------------------------------------------------------------

    def weak_phi_error(
        self,
        M_values: List[int],
        phi_functions: List[Callable],
        N_MC: int = 30,
        same_brownian: bool = False,
        error_type: str = "weak",
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """Weak (or strong) phi-observable error between Euler and Euler-with-RFF.

        Computes |E[phi(X_T)] - E[phi(X_T^M)]| for each phi and each M.

        Parameters
        ----------
        M_values      : list of RFF feature counts.
        phi_functions : list of callables phi: R -> R.
        N_MC          : number of Monte-Carlo paths.
        same_brownian : share Brownian path between schemes.
        error_type    : 'weak' or 'strong'.
        save_path     : optional Excel output.

        Returns
        -------
        error_vals : ndarray, shape (len(phi_functions), len(M_values))
        """
        n_phi    = len(phi_functions)
        error_vals = np.zeros((n_phi, len(M_values)))

        for j, M in enumerate(M_values):
            X_e_finals = []
            X_r_finals = []
            for _ in range(N_MC):
                G   = np.random.normal(size=self.N_t)
                bw  = G if same_brownian else None
                X_e = self.simulate("Euler", M=M, brownian_path=G)
                X_r = self.simulate("Euler with RFF", M=M, brownian_path=bw)
                X_e_finals.append(X_e[-1])
                X_r_finals.append(X_r[-1])

            X_e_arr = np.array(X_e_finals)
            X_r_arr = np.array(X_r_finals)

            for i, phi in enumerate(phi_functions):
                if error_type == "weak":
                    error_vals[i, j] = abs(
                        np.mean(phi(X_e_arr)) - np.mean(phi(X_r_arr))
                    )
                else:  # strong
                    error_vals[i, j] = float(
                        np.mean(np.abs(phi(X_e_arr) - phi(X_r_arr)))
                    )
            print(f"M={M} done.")

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, phi in enumerate(phi_functions):
            try:
                from RFFVolterraSfBM_Utils import lambda_to_latex_mapping
                label = lambda_to_latex_mapping(phi)
            except Exception:
                label = f"phi_{i}"
            ax.plot(M_values, error_vals[i], marker="o", label=label)

        ax.set_xlabel("M")
        if error_type == "weak":
            ax.set_ylabel(r"$|\mathbb{E}[\phi(X_T)] - \mathbb{E}[\phi(X_T^M)]|$")
        else:
            ax.set_ylabel(r"$\mathbb{E}[|\phi(X_T) - \phi(X_T^M)|]$")
        ax.legend(); ax.grid(True)
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path.replace(".xlsx", ".pdf"), bbox_inches="tight")
            rows = []
            for i in range(n_phi):
                for j, M in enumerate(M_values):
                    rows.append({"M": M, "phi_index": i, "error": error_vals[i, j]})
            pd.DataFrame(rows).to_excel(save_path, index=False)
            print(f"Saved: {save_path}")
        plt.close("all")
        return error_vals

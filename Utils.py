"""
===========================================================
File        : RFFVolterraSfBM_Utils.py
Project     : LogSfBM_Model
Authors     : Othmane Zarhali, Nicolas Langrené, Jean-François Muzy
Description :
    Standalone utility functions used by RFFVolterraSfBM_Class.py and
    the testing / example scripts.  Every function is a direct port of
    the corresponding standalone helper from the original script with
    identical behaviour, plus docstrings.

Contents
--------
    lambda_to_latex_mapping(phi, arg_name)
        Convert a Python lambda expression to a LaTeX label string.

    generate_grid(d, xmin, xmax, npts)
        Build a d-dimensional evaluation grid with 0 always included.

    FourierTransform_numeric(x, alpha, d)
        Numerical Fourier transform of (1 - ||u||^alpha) over the unit ball.

    FourierTransform_fromCloseForm(x, alpha, d)
        Closed-form hypergeometric Fourier transform.

    FourierTransformSfBMkernel_numeric(x, nu2, H, T, d, method)
        Numerical FT of the SfBM kernel at x.

    FourierTransformSfBMkernel_fromCloseForm(x, nu2, H, T, d)
        Closed-form FT of the SfBM kernel using hypergeometric functions.

    radial_pdf_on_r(r, nu, T, H, d)
        Unnormalized radial spectral density p_r(r).

    mc_phi(samples, u, return_samples)
        Monte-Carlo E[cos(eta^T u)] estimator.

    kernel_reconstruction_RFF(samples, u_list, nu2)
        Kernel reconstruction K^M(u) = (nu²/2) E[cos(eta^T u)].

    sample_from_density(n, nu, T, H, d, method, n_grid)
        Draw n samples from the SfBM spectral density (standalone version).
"""

from __future__ import annotations

import ast
import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union

import mpmath as mp
import numpy as np
from scipy import integrate, special
from scipy.integrate import quad
try:
    from scipy.integrate import simps
except ImportError:
    from scipy.integrate import simpson as simps
from scipy.special import jv, gamma as sp_gamma
from scipy.stats import f as f_dist, uniform

mp.mp.dps = 50


# ---------------------------------------------------------------------------
# Lambda → LaTeX label
# ---------------------------------------------------------------------------

def lambda_to_latex_mapping(phi, arg_name: str = "x") -> str:
    """Convert a lambda function to a LaTeX label string.

    Parses the source code of *phi* to extract its expression and
    translates it into a LaTeX string of the form ``$\\phi : x \\mapsto …$``.

    Parameters
    ----------
    phi      : callable (lambda)
    arg_name : str, variable name to use in the output label.

    Returns
    -------
    str  LaTeX label, or a fallback string when parsing fails.
    """
    try:
        src = inspect.getsource(phi)
    except Exception:
        return rf"$\phi : {arg_name} \mapsto \phi({arg_name})$"

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return rf"$\phi : {arg_name} \mapsto \phi({arg_name})$"

    lambda_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Lambda)]
    if not lambda_nodes:
        return rf"$\phi : {arg_name} \mapsto \phi({arg_name})$"

    # Identify the lambda that matches phi by comparing code constants/names
    target = None
    for node in lambda_nodes:
        try:
            tmp = eval(compile(ast.Expression(node), "<ast>", "eval"))
            if (tmp.__code__.co_consts == phi.__code__.co_consts and
                    tmp.__code__.co_names == phi.__code__.co_names):
                target = node
                break
        except Exception:
            continue

    if target is None:
        return rf"$\phi : {arg_name} \mapsto \phi({arg_name})$"

    def to_tex(n) -> str:
        if isinstance(n, ast.Name):
            return n.id
        if isinstance(n, ast.Constant):
            return str(n.value)
        if isinstance(n, ast.BinOp):
            if isinstance(n.op, ast.Pow):
                return rf"{to_tex(n.left)}^{{{to_tex(n.right)}}}"
            if isinstance(n.op, ast.Mult):
                return rf"{to_tex(n.left)} {to_tex(n.right)}"
            if isinstance(n.op, ast.Div):
                return rf"\frac{{{to_tex(n.left)}}}{{{to_tex(n.right)}}}"
            if isinstance(n.op, ast.Add):
                return rf"{to_tex(n.left)} + {to_tex(n.right)}"
            if isinstance(n.op, ast.Sub):
                return rf"{to_tex(n.left)} - {to_tex(n.right)}"
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return "-" + to_tex(n.operand)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            fname = n.func.id
            aarg  = to_tex(n.args[0])
            mapping = {"sin": r"\sin", "cos": r"\cos", "tan": r"\tan",
                       "log": r"\log", "exp": "e^", "sqrt": r"\sqrt"}
            if fname in ("sin", "cos", "tan", "log"):
                return rf"\{fname}({aarg})"
            if fname == "exp":
                return rf"e^{{{aarg}}}"
            if fname == "sqrt":
                return rf"\sqrt{{{aarg}}}"
            return rf"{fname}({aarg})"
        return "?"

    tex = to_tex(target.body)
    return rf"$\phi : {arg_name} \mapsto {tex}$"


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def generate_grid(
    d: int = 1,
    xmin: float = 1.0,
    xmax: float = 5.0,
    npts: int = 20,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Generate a d-dimensional evaluation grid with 0 always included.

    Parameters
    ----------
    d    : int    dimension.
    xmin : float  lower bound of each axis.
    xmax : float  upper bound of each axis.
    npts : int    number of points per axis.

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


# ---------------------------------------------------------------------------
# Spectral density: Fourier transform of (1 - ||u||^alpha) on the unit ball
# ---------------------------------------------------------------------------

def FourierTransform_numeric(
    x: Union[float, np.ndarray],
    alpha: float,
    d: int = 1,
) -> float:
    """Numerical Fourier transform of f(u) = (1-||u||^alpha) on ||u|| <= 1.

    Uses a radial Bessel reduction for any d.

    Parameters
    ----------
    x     : evaluation point (scalar or d-vector).
    alpha : exponent.
    d     : dimension.

    Returns
    -------
    float
    """
    x   = np.asarray(x, dtype=float)
    r   = float(np.linalg.norm(x))
    nu  = d / 2.0 - 1.0
    rtol_zero = 1e-12
    epsabs = epsrel = 1e-10

    if r <= rtol_zero:
        pref = 2.0 ** (-nu) / sp_gamma(nu + 1.0)
        return float(pref * (1.0 / d - 1.0 / (d + alpha)))

    def integrand(t):
        return (1.0 - t ** alpha) * (t ** (d / 2.0)) * special.jv(nu, r * t)

    I, _ = integrate.quad(integrand, 0.0, 1.0, epsabs=epsabs, epsrel=epsrel, limit=200)
    return float(I / (r ** nu))


def FourierTransform_fromCloseForm(
    x: Union[float, np.ndarray],
    alpha: float,
    d: int = 1,
) -> float:
    """Closed-form Fourier transform of (1-||u||^alpha) on ||u||<=1 via 0F1 and 1F2.

    Parameters
    ----------
    x     : evaluation point.
    alpha : exponent.
    d     : dimension.

    Returns
    -------
    float
    """
    x = np.asarray(x, dtype=float)
    r = float(np.linalg.norm(x))
    z = -(r ** 2) / 4.0

    prefactor = 1.0 / (2 ** (d / 2 - 1) * sp_gamma(d / 2))
    from mpmath import hyp0f1
    term1 = (1.0 / d) * float(hyp0f1(d / 2 + 1.0, z).real)
    a  = (d + alpha) / 2.0
    term2 = (1.0 / (d + alpha)) * float(mp.hyper([a], [d / 2.0, a + 1.0], z).real)
    return float(prefactor * (term1 - term2))


# ---------------------------------------------------------------------------
# SfBM kernel spectral density
# ---------------------------------------------------------------------------

def FourierTransformSfBMkernel_numeric(
    x: Union[float, np.ndarray],
    nu2: float,
    H: float,
    T: float,
    d: int = 1,
    method: str = "simpson",
) -> float:
    """Numerical Fourier transform of the SfBM kernel.

    K(tau) = (nu²/2)(1-(||tau||/T)^{2H}) for ||tau|| <= T.

    Parameters
    ----------
    x      : evaluation point.
    nu2, H, T : SfBM parameters.
    d      : dimension.
    method : integration method for d>=2 ('simpson').

    Returns
    -------
    float
    """
    x = np.asarray(x, dtype=float)
    if d == 1:
        scalar_x = float(x.ravel()[0])
        f = lambda u: nu2 / 2.0 * (1.0 - (abs(u) / T) ** (2.0 * H))
        val, _ = integrate.quad(f, 0, T, weight="cos", wvar=scalar_x,
                                epsabs=1e-12, epsrel=1e-12)
        return float(2.0 * val / (2.0 * math.pi) ** 0.5)

    k    = float(np.linalg.norm(x))
    nu_b = d / 2.0 - 1.0
    if k == 0.0:
        if d == 2:
            return (nu2 / 2.0) * (H / (2.0 * (1.0 + H))) * T ** 2
        return 0.0

    U    = k * T
    n_osc  = max(1.0, U / (2.0 * math.pi))
    N_base = int(math.ceil(n_osc * 20))
    if N_base % 2 == 1:
        N_base += 1

    prev, tol_rel, tol_abs = None, 1e-6, 1e-12
    I = 0.0
    for mult in [1, 2, 4, 8, 16, 32]:
        N   = N_base * mult
        u   = np.linspace(0.0, U, N + 1)
        hu  = (1.0 - (u / U) ** (2.0 * H)) * u ** (d / 2.0) * jv(nu_b, u)
        I   = (nu2 / 2.0) * simps(hu, u) / (k ** (d / 2.0 + 1.0))
        if prev is not None:
            err = abs(I - prev)
            if err / max(abs(I), tol_abs, abs(prev)) < tol_rel or err < tol_abs:
                break
        prev = I
    return float(I)


def FourierTransformSfBMkernel_fromCloseForm(
    x: Union[float, np.ndarray],
    nu2: float,
    H: float,
    T: float,
    d: int = 1,
) -> float:
    """Closed-form Fourier transform of the SfBM kernel via hypergeometric functions.

    f_omega(x) = pref * 1F2(d/2+H; d/2+H+1, d/2+1; -||x||²T²/4)

    Parameters
    ----------
    x      : evaluation point.
    nu2, H, T : SfBM parameters.
    d      : dimension.

    Returns
    -------
    float
    """
    x   = np.asarray(x, dtype=float)
    r   = float(np.linalg.norm(x))
    nu  = mp.sqrt(nu2)

    norm_x_sq = r ** 2
    pref = (
        float(nu) ** 2 * T ** d
        / (
            2 / (math.pi ** (d / 2))
            * (d / (2 * H) + 1)
            * sp_gamma(d / 2 + 1)
        )
    )
    z = -(norm_x_sq * T ** 2) / 4.0
    hg = mp.hyper([d / 2 + H], [d / 2 + H + 1, d / 2 + 1], z)
    return float((pref * hg).real)


# ---------------------------------------------------------------------------
# Radial density
# ---------------------------------------------------------------------------

def radial_pdf_on_r(
    r: float,
    nu: float,
    T: float,
    H: float,
    d: int = 1,
) -> float:
    """Unnormalized radial spectral density p_r(r) ∝ f_omega(r·e1) / (nu²/2).

    Parameters
    ----------
    r        : radial coordinate >= 0.
    nu       : sqrt(nu²).
    T, H     : SfBM parameters.
    d        : dimension.

    Returns
    -------
    float >= 0
    """
    nu2 = nu ** 2
    z   = -(T ** 2 * r ** 2) / 4.0
    pref  = (T ** 2 / 2) ** (d / 2) * mp.sqrt(nu2) ** 2 / mp.gamma(d / 2)
    term1 = (1 / d) * mp.hyper([], [d / 2 + 1], z)
    term2 = (1 / (d + 2 * H)) * mp.hyper([d / 2 + H], [d / 2, d / 2 + H + 1], z)
    val   = float((pref * (term1 - term2)).real)
    return max(val, 0.0) / (nu2 / 2.0)


# ---------------------------------------------------------------------------
# Monte-Carlo cosine expectation
# ---------------------------------------------------------------------------

def mc_phi(
    samples: np.ndarray,
    u: Union[np.ndarray, float],
    return_samples: bool = False,
) -> Union[float, np.ndarray]:
    """Monte-Carlo estimate of E[cos(eta^T u)].

    Parameters
    ----------
    samples       : (n, d) array of spectral samples.
    u             : evaluation point; shape (d,) or scalar for d=1.
    return_samples: if True, return per-sample cosine array (shape 1×n).

    Returns
    -------
    float or ndarray
    """
    u       = np.atleast_2d(u)           # (1, d)
    dots    = samples @ u.T              # (n, 1)
    cosvals = np.cos(dots)               # (n, 1)
    if return_samples:
        return cosvals.reshape(1, -1)
    return float(cosvals.mean())


def kernel_reconstruction_RFF(
    samples: np.ndarray,
    u_list: Union[np.ndarray, List],
    nu2: float,
) -> np.ndarray:
    """Reconstruct the SfBM kernel: K^M(u) = (nu²/2) * E[cos(eta^T u)].

    Parameters
    ----------
    samples : (n, d) spectral samples.
    u_list  : (n_pts, d) or (n_pts,) evaluation points.
    nu2     : nu² parameter.

    Returns
    -------
    ndarray, shape (n_pts,)
    """
    u_list = np.atleast_2d(u_list)
    K0     = nu2 / 2.0
    return np.array([K0 * mc_phi(samples, u) for u in u_list])


# ---------------------------------------------------------------------------
# Standalone sample_from_density (mirrors RFFVolterraSfBM.py original)
# ---------------------------------------------------------------------------

def sample_from_density(
    n_samples: int,
    nu: float,
    T: float,
    H: float,
    d: int = 1,
    n_grid: int = 3000,
    method: str = "Acceptance Rejection",
) -> np.ndarray:
    """Draw n_samples from the d-dimensional SfBM spectral density f_omega.

    Parameters
    ----------
    n_samples : int    number of samples.
    nu        : float  sqrt(nu²).
    T, H      : float  SfBM parameters.
    d         : int    dimension (1 or 2 for AR; any for HMC).
    n_grid    : int    grid size for CDF build (unused by HMC).
    method    : str    'Acceptance Rejection' or 'Hamiltonian MC'.

    Returns
    -------
    ndarray, shape (n_samples, d)
    """
    from RFFVolterraSfBM_Class import SpectralSampler
    sampler = SpectralSampler(
        nu2=nu ** 2, H=H, T=T, d=d,
        n_samples=n_samples, method=method,
    )
    return sampler.sample()


# ---------------------------------------------------------------------------
# Haar-wavelet pre-estimator (from RFFVolterraSfBM.py)
# ---------------------------------------------------------------------------

def EstH_RFF(
    xvals: np.ndarray,
    lagSig: List[int] = None,
) -> Tuple[float, float]:
    """Haar-wavelet pre-estimator of (H, lambda²) from a log-vol array.

    Parameters
    ----------
    xvals  : ndarray, shape (1, L) or (L,)  log-vol series.
    lagSig : list of int  lags to use (default: [1,2,4,8,16,32]).

    Returns
    -------
    (H, lambda2) : float, float
    """
    if lagSig is None:
        lagSig = [1, 2, 4, 8, 16, 32]
    lagSig = [k for k in lagSig if k >= 1]

    xvals = np.atleast_2d(xvals)
    xx1   = np.exp(xvals)
    xx1s  = np.cumsum(xx1, axis=1)
    zz    = np.zeros(len(lagSig))

    for i, scale in enumerate(lagSig):
        xxx      = xx1s[:, scale:] - xx1s[:, :-scale]
        xxx      = np.log(xxx)
        zz[i]   = np.mean(np.abs(xxx[:, scale:] - xxx[:, :-scale]))

    H  = np.polyfit(np.log(lagSig), np.log(zz + 1e-300), deg=1)[0]
    H  = max(H, 0.001)
    l2 = float((xvals[:, 1:] - xvals[:, :-1]).var() * 2.0 * H * (1.0 - 2.0 * H))
    return H, abs(l2)

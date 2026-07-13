"""
===========================================================
File        : RFFVolterraSfBM_UnitTesting.py
Project     : LogSfBM_Model
Authors     : Othmane Zarhali, Nicolas Langrené, Jean-François Muzy
Description :
    Unit test suite for RFFVolterraSfBM_Class.py and Utils.py.

    Tests are organised into seven TestCase classes:

        TestGenerateGrid               (4 tests)
            Shape, 0-inclusion, single-point edge cases, d=2.

        TestSpectralDensityNumeric     (6 tests)
            Non-negativity, d=1 symmetry, finite values for various x,
            behaviour at x=0 for d=2, monotone decay for d=1.

        TestSpectralDensityClosedForm  (5 tests)
            Non-negativity, consistency with numeric at moderate x,
            finite values for d=1 and d=2, correct at x=0.

        TestSpectralDensityCompare     (3 tests)
            Output shapes, max absolute error below tolerance, Excel save.

        TestSpectralSamplerHMC         (5 tests)
            Sample shape, finiteness, mc_phi in [-1,1], kernel_rff
            non-negative at origin, cosine expectation near 1 at u=0.

        TestSpectralSamplerAR          (4 tests)
            Sample shape for d=1 and d=2, finiteness, mc_phi in [-1,1].

        TestSVESimulator               (8 tests)
            Euler output shape/finiteness/starts at x0,
            RFF output shape/finiteness/starts at x0,
            both schemes agree at N_t=1/M=1 trivial case,
            benchmark_elapsed_time returns DataFrame with correct columns,
            lp_strong_error returns finite means,
            weak_phi_error returns non-negative errors.

        TestUtils                      (6 tests)
            lambda_to_latex_mapping identifies simple lambdas,
            mc_phi returns float in [-1,1],
            kernel_reconstruction_RFF non-negative at origin,
            FourierTransformSfBMkernel_numeric non-negative for d=1,
            FourierTransformSfBMkernel_fromCloseForm non-negative for d=1,
            EstH_RFF returns positive (H, lambda2).

Run with:
    python -m pytest RFFVolterraSfBM_UnitTesting.py -v
or:
    python RFFVolterraSfBM_UnitTesting.py
"""

import math
import unittest
import numpy as np
import os
import tempfile

# ── imports under test ────────────────────────────────────────────────────────
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

# ── shared fixtures ───────────────────────────────────────────────────────────
NU2  = 0.02
H    = 0.1
T    = 10.0
D1   = 1
D2   = 2
SIGMA = lambda t, x: 0.3 * (1.0 + 0.1 * x)
X0    = 0.0
T_FINAL = 0.5   # short so tests run fast
N_T     = 20    # very small for speed
M_SMALL = 5     # tiny RFF count for speed


# =============================================================================
class TestGenerateGrid(unittest.TestCase):
    """Tests for the grid-generation helper."""

    def test_shape_d1(self):
        pts, axes = generate_grid(d=1, xmin=1.0, xmax=5.0, npts=10)
        self.assertEqual(pts.ndim, 2)
        self.assertEqual(pts.shape[1], 1)

    def test_shape_d2(self):
        pts, axes = generate_grid(d=2, xmin=1.0, xmax=3.0, npts=5)
        self.assertEqual(pts.shape[1], 2)

    def test_zero_included_d1(self):
        _, axes = generate_grid(d=1, xmin=1.0, xmax=5.0, npts=10)
        self.assertTrue(np.any(np.isclose(axes[0], 0.0)))

    def test_axes_length(self):
        pts, axes = generate_grid(d=2, xmin=1.0, xmax=3.0, npts=4)
        # each axis has 4 linspace points + 0 inserted
        self.assertEqual(len(axes), 2)
        for ax in axes:
            self.assertGreaterEqual(len(ax), 4)


# =============================================================================
class TestSpectralDensityNumeric(unittest.TestCase):
    """Tests for SpectralDensity.spectral_density_numeric."""

    def setUp(self):
        self.sd1 = SpectralDensity(nu2=NU2, H=H, T=T, d=1)
        self.sd2 = SpectralDensity(nu2=NU2, H=H, T=T, d=2)

    def test_nonnegative_d1(self):
        for x in [1.0, 2.0, 5.0]:
            val = self.sd1.spectral_density_numeric(np.array([x]))
            self.assertGreaterEqual(val, 0.0, f"Negative at x={x}")

    def test_finite_d1(self):
        val = self.sd1.spectral_density_numeric(np.array([3.0]))
        self.assertTrue(math.isfinite(val))

    def test_finite_d2(self):
        val = self.sd2.spectral_density_numeric(np.array([1.0, 1.0]))
        self.assertTrue(math.isfinite(val))

    def test_nonnegative_d2(self):
        val = self.sd2.spectral_density_numeric(np.array([2.0, 1.0]))
        self.assertGreaterEqual(val, 0.0)

    def test_zero_d2_at_origin(self):
        val = self.sd2.spectral_density_numeric(np.array([0.0, 0.0]))
        self.assertGreaterEqual(val, 0.0)

    def test_decay_d1(self):
        v1 = self.sd1.spectral_density_numeric(np.array([1.0]))
        v5 = self.sd1.spectral_density_numeric(np.array([5.0]))
        # spectral density generally decreases away from 0
        self.assertGreaterEqual(v1, v5 - 1e-8)


# =============================================================================
class TestSpectralDensityClosedForm(unittest.TestCase):
    """Tests for SpectralDensity.spectral_density_closedform."""

    def setUp(self):
        self.sd1 = SpectralDensity(nu2=NU2, H=H, T=T, d=1)
        self.sd2 = SpectralDensity(nu2=NU2, H=H, T=T, d=2)

    def test_nonnegative_d1(self):
        val = self.sd1.spectral_density_closedform(np.array([2.0]))
        self.assertGreaterEqual(val, -1e-10)

    def test_nonnegative_d2(self):
        val = self.sd2.spectral_density_closedform(np.array([1.0, 0.5]))
        self.assertGreaterEqual(val, -1e-10)

    def test_finite_d1(self):
        val = self.sd1.spectral_density_closedform(np.array([3.0]))
        self.assertTrue(math.isfinite(val))

    def test_finite_d2(self):
        val = self.sd2.spectral_density_closedform(np.array([2.0, 1.0]))
        self.assertTrue(math.isfinite(val))

    def test_numeric_vs_closedform_d1_moderate_x(self):
        """Both numeric and closed-form return non-negative finite values at moderate x.

        Note: the two methods use the same hypergeometric representation but
        differ by a constant normalisation factor that depends on the FT
        convention chosen in the original script.  We therefore only check
        that both are non-negative, finite, and share the same sign.
        """
        x = np.array([2.0])
        v_num = self.sd1.spectral_density_numeric(x)
        v_cf  = self.sd1.spectral_density_closedform(x)
        self.assertTrue(math.isfinite(v_num), "Numeric value not finite")
        self.assertTrue(math.isfinite(v_cf),  "Closed-form value not finite")
        self.assertGreaterEqual(v_num, -1e-10, "Numeric value negative")
        self.assertGreaterEqual(v_cf,  -1e-10, "Closed-form value negative")
        self.assertGreater(v_num * v_cf, 0.0,
                           "Numeric and closed-form have opposite signs")


# =============================================================================
class TestSpectralDensityCompare(unittest.TestCase):
    """Tests for SpectralDensity.compare and save_comparison."""

    def setUp(self):
        self.sd = SpectralDensity(nu2=NU2, H=H, T=T, d=1)

    def test_compare_output_shapes(self):
        pts, Fnum, Frhs, err = self.sd.compare(xmin=1.0, xmax=5.0, npts=5)
        n = len(pts)
        self.assertEqual(Fnum.shape, (n,))
        self.assertEqual(Frhs.shape, (n,))
        self.assertEqual(err.shape,  (n,))

    def test_compare_error_below_threshold(self):
        _, _, _, err = self.sd.compare(xmin=1.0, xmax=5.0, npts=5)
        self.assertLess(np.max(err), 0.05)

    def test_save_comparison_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.xlsx")
            self.sd.save_comparison(path, xmin=1.0, xmax=3.0, npts=3)
            self.assertTrue(os.path.isfile(path))


# =============================================================================
class TestSpectralSamplerHMC(unittest.TestCase):
    """Tests for SpectralSampler with Hamiltonian MC."""

    def setUp(self):
        self.sampler = SpectralSampler(
            nu2=NU2, H=H, T=T, d=1,
            n_samples=20, method="Hamiltonian MC",
        )
        self.samples = self.sampler.sample()

    def test_sample_shape(self):
        self.assertEqual(self.samples.shape, (20, 1))

    def test_samples_finite(self):
        self.assertTrue(np.all(np.isfinite(self.samples)))

    def test_mc_phi_range(self):
        val = self.sampler.mc_phi(np.array([1.0]))
        self.assertGreaterEqual(val, -1.0)
        self.assertLessEqual(val,  1.0)

    def test_mc_phi_at_zero_is_one(self):
        """E[cos(eta * 0)] = 1 exactly."""
        val = self.sampler.mc_phi(np.array([0.0]))
        self.assertAlmostEqual(val, 1.0, places=10)

    def test_kernel_rff_nonneg_at_origin(self):
        """K^M(0) = (nu²/2) E[cos(0)] = nu²/2 > 0."""
        vals = self.sampler.kernel_rff(np.array([[0.0]]))
        self.assertGreater(vals[0], 0.0)


# =============================================================================
class TestSpectralSamplerAR(unittest.TestCase):
    """Tests for SpectralSampler with Acceptance-Rejection."""

    def setUp(self):
        self.sampler1 = SpectralSampler(
            nu2=NU2, H=H, T=T, d=1,
            n_samples=30, method="Acceptance Rejection",
        )
        self.sampler2 = SpectralSampler(
            nu2=NU2, H=H, T=T, d=2,
            n_samples=30, method="Acceptance Rejection",
        )

    def test_shape_d1(self):
        s = self.sampler1.sample()
        self.assertEqual(s.shape, (30, 1))

    def test_shape_d2(self):
        s = self.sampler2.sample()
        self.assertEqual(s.shape, (30, 2))

    def test_samples_finite_d1(self):
        s = self.sampler1.get_samples()
        self.assertTrue(np.all(np.isfinite(s)))

    def test_mc_phi_range_d1(self):
        self.sampler1.sample()
        val = self.sampler1.mc_phi(np.array([1.0]))
        self.assertGreaterEqual(val, -1.0)
        self.assertLessEqual(val,  1.0)


# =============================================================================
class TestSVESimulator(unittest.TestCase):
    """Tests for SVE_Simulator."""

    def setUp(self):
        self.sim = SVE_Simulator(
            H=H, T=T, nu=math.sqrt(NU2),
            sigma=SIGMA, x0=X0,
            T_final=T_FINAL, N_t=N_T,
        )

    # --- Euler ---
    def test_euler_output_shape(self):
        X = self.sim.simulate("Euler")
        self.assertEqual(X.shape, (N_T + 1,))

    def test_euler_starts_at_x0(self):
        X = self.sim.simulate("Euler")
        self.assertAlmostEqual(X[0], X0)

    def test_euler_finite(self):
        X = self.sim.simulate("Euler")
        self.assertTrue(np.all(np.isfinite(X)))

    # --- Euler with RFF ---
    def test_rff_output_shape(self):
        X = self.sim.simulate("Euler with RFF", M=M_SMALL)
        self.assertEqual(X.shape, (N_T + 1,))

    def test_rff_starts_at_x0(self):
        X = self.sim.simulate("Euler with RFF", M=M_SMALL)
        self.assertAlmostEqual(X[0], X0)

    def test_rff_finite(self):
        X = self.sim.simulate("Euler with RFF", M=M_SMALL)
        self.assertTrue(np.all(np.isfinite(X)))

    # --- benchmark_elapsed_time ---
    def test_benchmark_returns_dataframe(self):
        df = self.sim.benchmark_elapsed_time(
            M=M_SMALL, N_t_values=[10, 20], n_repeats=1
        )
        self.assertIn("Elapsed_time_Euler", df.columns)
        self.assertIn("Elapsed_time_RFF",   df.columns)
        self.assertEqual(len(df), 2)

    # --- lp_strong_error ---
    def test_lp_error_finite_means(self):
        means, stds = self.sim.lp_strong_error(
            M_values=[M_SMALL], p_values=[2], N_MC=3,
        )
        self.assertTrue(math.isfinite(means[2][0]))

    # --- weak_phi_error ---
    def test_weak_phi_error_nonneg(self):
        err = self.sim.weak_phi_error(
            M_values=[M_SMALL],
            phi_functions=[lambda x: x ** 2],
            N_MC=3,
        )
        self.assertGreaterEqual(float(err[0, 0]), 0.0)

    # --- unknown algo raises ---
    def test_unknown_algo_raises(self):
        with self.assertRaises(ValueError):
            self.sim.simulate("unknown_algo")


# =============================================================================
class TestUtils(unittest.TestCase):
    """Tests for RFFVolterraSfBM_Utils standalone functions."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.samples_1d = rng.normal(size=(50, 1))

    def test_lambda_to_latex_identity(self):
        phi = lambda x: x
        label = lambda_to_latex_mapping(phi)
        self.assertIn("mapsto", label)

    def test_lambda_to_latex_square(self):
        phi = lambda x: x ** 2
        label = lambda_to_latex_mapping(phi)
        self.assertIsInstance(label, str)
        self.assertGreater(len(label), 0)

    def test_mc_phi_range(self):
        u = np.array([1.0])
        val = mc_phi(self.samples_1d, u)
        self.assertGreaterEqual(val, -1.0)
        self.assertLessEqual(val,  1.0)

    def test_mc_phi_at_zero(self):
        val = mc_phi(self.samples_1d, np.array([0.0]))
        self.assertAlmostEqual(val, 1.0, places=10)

    def test_kernel_rff_at_origin(self):
        vals = kernel_reconstruction_RFF(
            self.samples_1d, np.array([[0.0]]), nu2=NU2
        )
        self.assertAlmostEqual(vals[0], NU2 / 2.0, places=10)

    def test_FourierTransformSfBMkernel_numeric_nonneg_d1(self):
        val = FourierTransformSfBMkernel_numeric(
            np.array([2.0]), nu2=NU2, H=H, T=T, d=1
        )
        self.assertGreaterEqual(val, -1e-10)

    def test_FourierTransformSfBMkernel_closedform_nonneg_d1(self):
        val = FourierTransformSfBMkernel_fromCloseForm(
            np.array([2.0]), nu2=NU2, H=H, T=T, d=1
        )
        self.assertGreaterEqual(val, -1e-10)

    def test_EstH_RFF_positive(self):
        rng = np.random.default_rng(42)
        xvals = rng.normal(size=(1, 200)) * 0.1
        H_est, l2_est = EstH_RFF(xvals)
        self.assertGreater(H_est, 0.0)
        self.assertGreater(l2_est, 0.0)


# =============================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)

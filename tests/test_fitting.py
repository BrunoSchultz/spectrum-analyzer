import unittest
import numpy as np
import matplotlib.pyplot as plt

from spectrum_analyzer.fitting import (
    fit_multiple_peaks,
    gaussian,
    lorentzian,
    voigt
)

class TestFitting(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(0, 10, 1000)
        self.noise_level = 0.2

        self.n_params_map = {
            'gaussian': 3,
            'lorentzian': 3,
            'voigt': 4
        }

    def _generate_data(self, peak_type, params, n_peaks, noise_level=None):
        y = np.zeros_like(self.x)
        n_params = self.n_params_map[peak_type]
        for i in range(n_peaks):
            start = i * n_params
            peak_params = params[start:start + n_params]
            if peak_type == 'gaussian':
                y += gaussian(self.x, *peak_params)
            elif peak_type == 'lorentzian':
                y += lorentzian(self.x, *peak_params)
            elif peak_type == 'voigt':
                y += voigt(self.x, *peak_params)
        noise = np.random.normal(0, noise_level if noise_level is not None else self.noise_level, size=len(self.x))
        y += noise
        return y

    def _plot_fit(self, y_data, popt, peak_type, n_peaks, title):
        n_params = self.n_params_map[peak_type]
        y_fit = np.zeros_like(self.x)
        for i in range(n_peaks):
            start = i * n_params
            peak_params = popt[start:start + n_params]
            if peak_type == 'gaussian':
                y_fit += gaussian(self.x, *peak_params)
            elif peak_type == 'lorentzian':
                y_fit += lorentzian(self.x, *peak_params)
            elif peak_type == 'voigt':
                y_fit += voigt(self.x, *peak_params)
        plt.figure(figsize=(8, 4))
        plt.plot(self.x, y_data, label='Noisy data')
        plt.plot(self.x, y_fit, label='Fitted peaks', linewidth=2)
        plt.title(title)
        plt.legend()
        plt.show()

    def _run_test(self, peak_type, n_peaks, true_params, p0, noise_level=None, delta=0.5):
        y = self._generate_data(peak_type, true_params, n_peaks, noise_level=noise_level)
        popt, _ = fit_multiple_peaks(self.x, y, peak_type, n_peaks, p0=p0)

        for true_val, fitted_val in zip(true_params, popt):
            self.assertAlmostEqual(true_val, fitted_val, delta=delta)

        self._plot_fit(y, popt, peak_type, n_peaks, f"Fit for {n_peaks} {peak_type} peaks")

    # --- Single Peak Tests --- #

    def test_single_gaussian_peak(self):
        peak_type = 'gaussian'
        n_peaks = 1
        true_params = [5, 3.0, 0.3]
        p0 = [4.5, 2.9, 0.25]
        self._run_test(peak_type, n_peaks, true_params, p0)

    def test_single_lorentzian_peak(self):
        peak_type = 'lorentzian'
        n_peaks = 1
        true_params = [5, 3.0, 0.3]
        p0 = [4.5, 2.9, 0.25]
        self._run_test(peak_type, n_peaks, true_params, p0)

    def test_single_voigt_peak(self):
        peak_type = 'voigt'
        n_peaks = 1
        true_params = [5, 3.0, 0.3, 0.15]
        p0 = [4.5, 2.9, 0.25, 0.12]
        self._run_test(peak_type, n_peaks, true_params, p0)

    # --- Multiple Peaks Tests --- #

    def test_multiple_gaussian_peaks(self):
        peak_type = 'gaussian'
        n_peaks = 3
        true_params = [
            5, 2.5, 0.3,
            3, 5.0, 0.4,
            4, 7.5, 0.2
        ]
        p0 = [
            4, 2.4, 0.25,
            2.5, 4.8, 0.35,
            3.8, 7.3, 0.18
        ]
        self._run_test(peak_type, n_peaks, true_params, p0)

    def test_multiple_lorentzian_peaks(self):
        peak_type = 'lorentzian'
        n_peaks = 3
        true_params = [
            5, 2.5, 0.3,
            3, 5.0, 0.4,
            4, 7.5, 0.2
        ]
        p0 = [
            4, 2.4, 0.25,
            2.5, 4.8, 0.35,
            3.8, 7.3, 0.18
        ]
        self._run_test(peak_type, n_peaks, true_params, p0)

    def test_multiple_voigt_peaks(self):
        peak_type = 'voigt'
        n_peaks = 3
        true_params = [
            5, 2.5, 0.3, 0.15,
            3, 5.0, 0.4, 0.2,
            4, 7.5, 0.2, 0.1
        ]
        p0 = [
            4, 2.4, 0.25, 0.1,
            2.5, 4.8, 0.35, 0.15,
            3.8, 7.3, 0.18, 0.08
        ]
        self._run_test(peak_type, n_peaks, true_params, p0)

    # --- Overlapping Peaks with High Noise --- #

    def test_overlapping_peaks_high_noise_all_types(self):
        n_peaks = 3
        noise_level = 5  # very high noise
        delta = 5        # loose tolerance due to difficulty

        # VERY CLOSE and overlapping peak centers
        centers = [3.0, 3.5, 4]

        # Gaussian params: amp, mean, stddev
        true_params_gauss = []
        p0_gauss = []
        for c in centers:
            true_params_gauss.extend([5, c, 0.4])
            p0_gauss.extend([4, c - 0.1, 0.35])

        self._run_test('gaussian', n_peaks, true_params_gauss, p0_gauss, noise_level=noise_level, delta=delta)

        # Lorentzian params: amp, mean, gamma
        true_params_lorentz = []
        p0_lorentz = []
        for c in centers:
            true_params_lorentz.extend([5, c, 0.4])
            p0_lorentz.extend([4, c - 0.1, 0.35])

        self._run_test('lorentzian', n_peaks, true_params_lorentz, p0_lorentz, noise_level=noise_level, delta=delta)

        # Voigt params: amp, mean, sigma, gamma
        true_params_voigt = []
        p0_voigt = []
        for c in centers:
            true_params_voigt.extend([5, c, 0.4, 0.15])
            p0_voigt.extend([4, c - 0.1, 0.35, 0.1])

        self._run_test('voigt', n_peaks, true_params_voigt, p0_voigt, noise_level=noise_level, delta=delta)

if __name__ == "__main__":
    unittest.main()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import wofz


# Define peak profiles
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))

def lorentzian(x, amp, cen, wid):
    return amp * wid**2 / ((x - cen)**2 + wid**2)

def voigt(x, amp, cen, sigma, gamma):
    z = ((x - cen) + 1j * gamma) / (sigma * np.sqrt(2))
    return amp * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

# Define background functions
def background(x, kind='constant', coeffs=None):
    if kind == 'constant':
        return coeffs[0] * np.ones_like(x)
    elif kind == 'linear':
        return coeffs[0] + coeffs[1] * x
    elif kind == 'cubic':
        return np.polyval(coeffs, x)
    else:
        raise ValueError("Unsupported background type.")

# Model generator
def make_model(x, peak_type, n_peaks, bg_type, bg_coeffs):
    def model(x, *params):
        bg = background(x, bg_type, bg_coeffs)
        result = np.zeros_like(x, dtype=np.float64)
        for i in range(n_peaks):
            if peak_type == 'gaussian':
                result += gaussian(x, *params[i*3:i*3+3])
            elif peak_type == 'lorentzian':
                result += lorentzian(x, *params[i*3:i*3+3])
            elif peak_type == 'voigt':
                result += voigt(x, *params[i*4:i*4+4])
            else:
                raise ValueError("Unsupported peak type.")
        return result + bg
    return model

# Fit function
def fit_peaks(x, y, peak_type='gaussian', n_peaks=1, bg_type='constant', bg_coeffs=[0.0], initial_params=None):
    """
    Fit multiple peaks to the data.

    Parameters
    ----------
    x : array-like
        The x values of the spectrum.
    y : array-like
        The y values of the spectrum.
    peak_type : str
        The type of peak to fit ('gaussian', 'lorentzian', 'voigt').
    n_peaks : int
        The number of peaks to fit.
    bg_type : str
        The type of background to fit ('constant', 'linear', 'cubic').
    bg_coeffs : list
        The coefficients for the background polynomial.
    initial_params : list, optional
        Initial parameters for the fit. If None, defaults will be generated.
        For 'gaussian' and 'lorentzian': [amp1, cen1, wid1, ..., ampN, cenN, widN]
        For 'voigt': [amp1, cen1, sigma1, gamma1, ..., ampN, cenN, sigmaN, gammaN]

    Returns
    -------
    popt : array
        The optimal parameters found for the fit.
    pcov : 2D array
        The covariance of the optimal parameters.
    y_fit : array
        The fitted curve evaluated at x.
    """
    model = make_model(x, peak_type, n_peaks, bg_type, bg_coeffs)
    if initial_params is None:
        if peak_type in ['gaussian', 'lorentzian']:
            initial_params = []
            for i in range(n_peaks):
                amp = max(y)
                cen = x[len(x)//(n_peaks+1)*(i+1)]
                wid = (max(x) - min(x)) / (4 * n_peaks)
                initial_params += [amp, cen, wid]
        elif peak_type == 'voigt':
            initial_params = []
            for i in range(n_peaks):
                amp = max(y)
                cen = x[len(x)//(n_peaks+1)*(i+1)]
                sigma = (max(x) - min(x)) / (6 * n_peaks)
                gamma = sigma
                initial_params += [amp, cen, sigma, gamma]

    # First fit
    popt, _ = curve_fit(model, x, y, p0=initial_params, maxfev=10000)

    # Second fit using first fit results
    popt_refined, pcov_refined = curve_fit(model, x, y, p0=popt)

    return popt_refined, pcov_refined, model(x, *popt_refined)

# Example usage
if __name__ == '__main__':
    # Generate synthetic data
    x = np.linspace(-10, 10, 1000)
    y_true = gaussian(x, 9, 2, 1) + gaussian(x, 8, 4, 1.5) + background(x, 'cubic', [0.02, 0.01, 0.01])
    noise = np.random.normal(0, 0.5, x.size)
    y = y_true + noise

    # Fit
    popt, _, y_fit = fit_peaks(x, y, peak_type='gaussian', n_peaks=2, bg_type='cubic', bg_coeffs=[0.02, 0.01, 0.01], initial_params=[9.1, 2.4, 1.1, 8.5, 4.1, 2])

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Data', alpha=0.6)
    plt.plot(x, y_fit, label='Fit', lw=2)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Multi-Peak Fitting")
    plt.show()

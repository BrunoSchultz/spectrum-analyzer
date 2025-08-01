import numpy as np
from scipy.optimize import curve_fit
from scipy.special import voigt_profile


# --- Peak profiles --- #


def gaussian(x, amp, mean, stddev):
    """
    Gaussian function for peak fitting.
    
    Parameters:
    x : array-like
        The x values.
    amp : float
        Amplitude of the Gaussian.
    mean : float
        Mean (center) of the Gaussian.
    stddev : float
        Standard deviation of the Gaussian.
    
    Returns:
    array-like
        The Gaussian function evaluated at x.
    """
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))


def lorentzian(x, amp, mean, gamma):
    """
    Lorentzian function for peak fitting.
    
    Parameters:
    x : array-like
        The x values.
    amp : float
        Amplitude of the Lorentzian.
    mean : float
        Mean (center) of the Lorentzian.
    gamma : float
        Half-width at half-maximum (HWHM) of the Lorentzian.
    
    Returns:
    array-like
        The Lorentzian function evaluated at x.
    """
    return amp * (gamma**2 / ((x - mean)**2 + gamma**2))


def voigt(x, amp, mean, sigma, gamma):
    """
    Voigt profile for peak fitting.
    
    Parameters:
    x : array-like
        The x values.
    amp : float
        Amplitude of the Voigt profile.
    mean : float
        Mean (center) of the Voigt profile.
    sigma : float
        Standard deviation of the Gaussian component.
    gamma : float
        Half-width at half-maximum (HWHM) of the Lorentzian component.
    
    Returns:
    array-like
        The Voigt profile evaluated at x.
    """
    return amp * voigt_profile(x - mean, sigma, gamma)


# --- Composite peak fitting --- #


def fit_single_peak(x, y, peak_type, p0=None, bounds=None):
    """
    Fit a single peak to the data.
    
    Parameters:
    x : array-like
        The x values.
    y : array-like
        The y values.
    peak_type : str
        Type of peak ('gaussian', 'lorentzian', 'voigt').
    p0 : list, optional
        Initial guess for the parameters.
    bounds : tuple, optional
        Bounds for the parameters.
    
    Returns:
    popt : array
        Optimal values for the parameters.
    pcov : 2D array
        Covariance of popt.
    """
    if peak_type == 'gaussian':
        func = gaussian
    elif peak_type == 'lorentzian':
        func = lorentzian
    elif peak_type == 'voigt':
        func = voigt
    else:
        raise ValueError("Unsupported peak type: {}".format(peak_type))
    
    popt, pcov = curve_fit(func, x, y, p0=p0, bounds=bounds)

    return popt, pcov


def num_params(peak_type):
    return {
        'gaussian': 3,
        'lorentzian': 3,
        'voigt': 4
    }.get(peak_type, None)

def fit_multiple_peaks(x, y, peak_type, n_peaks, p0=None, bounds=None):
    """
    Fit multiple peaks of the same type to the data.

    Parameters:
    x : array-like
        The x values.
    y : array-like
        The y values.
    peak_type : str
        One of 'gaussian', 'lorentzian', or 'voigt'.
    n_peaks : int
        Number of peaks to fit.
    p0 : list, optional
        Initial guess for parameters.
    bounds : tuple of lists, optional
        Bounds for parameters: (lower_bounds, upper_bounds)

    Returns:
    popt : array
        Optimal parameters (flattened).
    pcov : 2D array
        Covariance of popt.
    """

    n_params = num_params(peak_type)
    if n_params is None:
        raise ValueError(f"Unsupported peak type: {peak_type}")

    def single_peak(x, *params):
        if peak_type == 'gaussian':
            return gaussian(x, *params)
        elif peak_type == 'lorentzian':
            return lorentzian(x, *params)
        elif peak_type == 'voigt':
            return voigt(x, *params)
        else:
            raise ValueError(f"Unsupported peak type: {peak_type}")

    def composite_model(x, *params):
        y_total = np.zeros_like(x)
        for i in range(n_peaks):
            start = i * n_params
            peak_params = params[start:start + n_params]
            y_total += single_peak(x, *peak_params)
        return y_total

    expected_param_len = n_peaks * n_params

    if p0 is not None and len(p0) != expected_param_len:
        raise ValueError(f"Expected {expected_param_len} parameters in p0, got {len(p0)}.")

    if bounds is not None:
        if len(bounds) != 2:
            raise ValueError("Bounds must be a tuple of (lower_bounds, upper_bounds)")
        if len(bounds[0]) != expected_param_len or len(bounds[1]) != expected_param_len:
            raise ValueError("Each bound list must have length equal to total number of parameters.")

    if bounds is None:
        popt, pcov = curve_fit(composite_model, x, y, p0=p0)
    else:
        popt, pcov = curve_fit(composite_model, x, y, p0=p0, bounds=bounds)

    return popt, pcov

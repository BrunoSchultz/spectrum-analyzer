import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import string
from fitting import fit_peaks
import warnings
from scipy.optimize import OptimizeWarning
import scienceplots
plt.style.use(['science', 'no-latex'])  # prettier scientific style

class FitResult:
    def __init__(self, params, errors, peak_type, n_peaks, bg_type, bg_coeffs, fit_index, x_min=None, x_max=None):
        """
        Create a fitting result for one multi-peak fit.

        Parameters
        ----------s
        params : list or array
            Optimized parameters from fitting.
        errors : list or array
            Errors from the covariance matrix.
        peak_type : str
            Type of peak ('gaussian', etc.)
        n_peaks : int
            Number of peaks fitted together.
        bg_type : str
            Type of background.
        bg_coeffs : list
            Background coefficients.
        fit_index : int
            Sequential index used for naming (e.g., 1, 2a, 2b)
        """
        self.df = self._format_result(params, errors, peak_type, n_peaks, bg_type, bg_coeffs, fit_index, x_min, x_max)
        self.x_min = x_min
        self.x_max = x_max

    def _format_result(self, params, errors, peak_type, n_peaks, bg_type, bg_coeffs, fit_index, x_min, x_max):
        if peak_type in ['gaussian', 'lorentzian']:
            group_size = 3
            columns = ['amp', 'cen', 'wid']
        elif peak_type == 'voigt':
            group_size = 4
            columns = ['amp', 'cen', 'sigma', 'gamma']
        else:
            raise ValueError("Unsupported peak type.")

        result_rows = []
        for i in range(n_peaks):
            base_index = i * group_size
            param_set = params[base_index:base_index + group_size]
            error_set = errors[base_index:base_index + group_size]
            label = str(fit_index) + (string.ascii_lowercase[i] if n_peaks > 1 else '')

            row = {
                'peak_id': label,
                'type': peak_type,
                'bg_type': bg_type,
                'bg_coeffs': bg_coeffs,
                'x_min': x_min,
                'x_max': x_max
            }
            # Add parameters
            row.update(dict(zip(columns, param_set)))
            # Add errors with 'd_' prefix
            error_columns = ['d_' + c for c in columns]
            row.update(dict(zip(error_columns, error_set)))

            result_rows.append(row)

        return pd.DataFrame(result_rows)
    
    def __repr__(self):
        """
        Return a concise DataFrame-style representation of the fit result.
        """
        return self.as_dataframe().to_string()

    def as_dataframe(self, full=False):
        if full:
            return self.df
        else:
            # Hide background and error columns unless full=True
            return self.df.drop(columns=['bg_type', 'bg_coeffs'], errors='ignore')


class Spectrum:
    def __init__(self, filepath=None):
        self.x = None
        self.y = None
        self.filepath = filepath
        self.fitted_peaks = pd.DataFrame(columns=[
            'peak_id', 'type', 'amp', 'cen', 'wid', 'sigma', 'gamma',
            'bg_type', 'bg_coeffs'
        ])
        if filepath:
            self.load_spectrum(filepath)

        self._fit_count = 1  # for unique fit_index tracking


    def __repr__(self): 
        """
        Return a concise DataFrame-style representation of the spectrum data.
        """
        if self.x is None or self.y is None:
            return "<Spectrum: No data loaded>"

        df_preview = pd.DataFrame({'x': self.x, 'y': self.y})
        return f"<Spectrum: {len(df_preview)} points>\n" + repr(df_preview.head(10))


    def load_spectrum(self, filepath, delimiter=None, x_col=None, y_col=None, skiprows=0, header='infer'):
        """
        Robustly load spectrum from a CSV file.

        Parameters
        ----------
        filepath : str
            Path to the CSV file.
        delimiter : str, optional
            Delimiter for CSV (e.g., ',', '\t', ';'). If None, tries to auto-detect.
        x_col : str or int, optional
            Name or index of the x column.
        y_col : str or int, optional
            Name or index of the y column.
        skiprows : int
            Rows to skip at the start of the file.
        header : int, list of int, 'infer', or None
            Row number(s) to use as the column names.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"{filepath} not found.")

        # Auto-detect delimiter if not specified
        if delimiter is None:
            with open(filepath, 'r') as f:
                sample = f.read(1024)
                sniffer = csv.Sniffer()
                try:
                    dialect = sniffer.sniff(sample)
                    delimiter = dialect.delimiter
                except csv.Error:
                    delimiter = ','  # Default fallback

        # Load the data
        try:
            data = pd.read_csv(filepath, delimiter=delimiter, skiprows=skiprows, header=header)
        except Exception as e:
            raise ValueError(f"Failed to read CSV: {e}")

        # Determine columns
        if x_col is None or y_col is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                raise ValueError("CSV must contain at least two numeric columns for x and y.")
            x_col, y_col = numeric_cols[:2]

        try:
            self.x = data[x_col].values
            self.y = data[y_col].values
        except KeyError:
            raise ValueError(f"Specified columns '{x_col}' or '{y_col}' not found in the file.")

        self.filepath = filepath
        print(f"Spectrum loaded from {filepath} using delimiter='{delimiter}'.")

    def get_segment(self, x_min=None, x_max=None, inplace=False):
        """
        Returns a new Spectrum object containing only the selected range, or modifies in place.

        Parameters
        ----------
        x_min : float or None
            Minimum x-value of the segment.
        x_max : float or None
            Maximum x-value of the segment.
        inplace : bool
            If True, modifies the current Spectrum object in place. If False, returns a new Spectrum.

        Returns
        -------
        Spectrum or None
            A Spectrum instance containing the selected data if inplace=False, otherwise None.
        """
        if self.x is None or self.y is None:
            raise ValueError("Spectrum data not loaded.")
        
        mask = np.ones_like(self.x, dtype=bool)
        if x_min is not None:
            mask &= self.x >= x_min
        if x_max is not None:
            mask &= self.x <= x_max

        if inplace:
            self.x = self.x[mask]
            self.y = self.y[mask]
            self.filepath = (self.filepath or "") + f" [Segment {x_min}-{x_max}]"
            return None
        else:
            segment = Spectrum()
            segment.x = self.x[mask]
            segment.y = self.y[mask]
            segment.filepath = (self.filepath or "") + f" [Segment {x_min}-{x_max}]"
            return segment

    def add_fitted_peak(self, fit_result):
        """
        Add a FitResult object's DataFrame to the fitted_peaks registry.

        Parameters
        ----------
        fit_result : FitResult
            The result object from a peak fitting.
        """
        self.fitted_peaks = pd.concat([self.fitted_peaks, fit_result.df], ignore_index=True)

    def remove_fitted_peak(self, peak_id):
        """
        Remove a fitted peak by its ID.

        Parameters
        ----------
        peak_id : str
            The ID of the peak to remove.
        """
        if peak_id in self.fitted_peaks['peak_id'].values:
            self.fitted_peaks = self.fitted_peaks[self.fitted_peaks['peak_id'] != peak_id]
        else:
            raise ValueError(f"Peak ID '{peak_id}' not found in fitted peaks.")

    def fit_peaks(self, x_min=None, x_max=None, peak_type='gaussian', n_peaks=1, bg_type='constant', bg_coeffs=[0.0], initial_params=None):
        x, y = self.x, self.y

        if x_min is not None or x_max is not None:
            segment = self.get_segment(x_min, x_max, inplace=False)
            if segment is None:
                raise ValueError("No data available in the specified range.")
            x, y = segment.x, segment.y

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", OptimizeWarning)  # Catch OptimizeWarning
            
            popt, pcov, _ = fit_peaks(x, y, peak_type=peak_type, n_peaks=n_peaks, bg_type=bg_type, bg_coeffs=bg_coeffs, initial_params=initial_params)

            # Check if an OptimizeWarning was raised
            if any(issubclass(warning.category, OptimizeWarning) for warning in w):
                print("Warning: Covariance of the parameters could not be estimated. Fit result not added.")
                return None  # Or raise, or handle differently

        errors = np.sqrt(np.diag(pcov))
        fit_index = self._fit_count
        self._fit_count += 1

        fit_result = FitResult(popt, errors, peak_type, n_peaks, bg_type, bg_coeffs, fit_index, x_min, x_max)

        self.add_fitted_peak(fit_result)

        return fit_result

    def export_fitted_peaks(self, out_path):
        """
        Export the fitted peaks data to a CSV.
        """
        self.fitted_peaks.to_csv(out_path, index=False)
        print(f"Fitted peak data exported to {out_path}")


    def plot(self, x_min=None, x_max=None, 
             title=None, xlabel="$x$", ylabel="$y$",
             color='blue', lw=1.5, 
             show=True, save_path=None, dpi=700,
             legend=False, figsize=(6, 4),
             ax=None, fig=None,
             **kwargs):
        """
        Plot the full spectrum or a selected segment with customizable styling.

        Parameters
        ----------
        x_min, x_max : float or None
            Limits of x data to plot.
        title : str or None
            Plot title. Default None (no title).
        xlabel, ylabel : str
            Axis labels.
        color : str
            Line color.
        lw : float
            Line width.
        show : bool
            Whether to show the plot interactively.
        save_path : str or None
            File path to save figure. If None, no file saved.
        dpi : int
            Resolution of saved figure.
        legend : bool
            Whether to show legend. Default False.
        figsize : tuple
            Figure size in inches.
        ax : matplotlib.axes.Axes or None
            Axes to plot on; if None, creates new figure and axes.
        fig : matplotlib.figure.Figure or None
            Figure instance; only used if ax is None.
        kwargs : dict
            Additional keyword args passed to plt.plot()
        """

        if self.x is None or self.y is None:
            raise ValueError("No data loaded in the Spectrum object.")

        segment = self.get_segment(x_min, x_max)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        ax.plot(segment.x, segment.y, color=color, lw=lw, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)

        if title is not None:
            ax.set_title(title)

        if legend:
            ax.legend()

        if save_path:
            fig.savefig(save_path, dpi=dpi)
            print(f"Plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)


    def plot_fit_overlay(self, x_min=None, x_max=None,
                         show_individual_peaks=True,
                         show_total_fit=True,
                         title=None, xlabel="$x$", ylabel="$y$",
                         colors=None,
                         figsize=(6, 4), dpi=700,
                         save_path=None, show=True,
                         legend=False,
                         fig=None, ax=None,
                         **kwargs):
        """
        Plot the spectrum with fitted peak overlays, customizable with keyword arguments.

        Parameters
        ----------
        x_min, x_max : float or None
            Plot x limits.
        show_individual_peaks : bool
            Plot each individual peak.
        show_total_fit : bool
            Plot the total fit sum.
        title : str or None
            Plot title.
        xlabel, ylabel : str
            Axis labels.
        colors : matplotlib colormap or None
            Colormap for peaks.
        figsize : tuple
            Figure size.
        dpi : int
            Save dpi.
        save_path : str or None
            File path to save figure.
        show : bool
            Show figure interactively.
        legend : bool
            Show legend.
        fig : matplotlib.figure.Figure or None
            Figure to plot on.
        ax : matplotlib.axes.Axes or None
            Axes to plot on.
        kwargs : dict
            Extra kwargs passed to plot calls.
        """
        if self.x is None or self.y is None:
            raise ValueError("No data loaded in the Spectrum object.")
        if self.fitted_peaks.empty:
            raise ValueError("No fitted peaks to overlay.")

        # Prepare axes
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Mask for plot range
        mask = np.ones_like(self.x, dtype=bool)
        if x_min is not None:
            mask &= self.x >= x_min
        if x_max is not None:
            mask &= self.x <= x_max

        x_base = self.x[mask]
        y_base = self.y[mask]

        ax.plot(x_base, y_base, label="Original Spectrum", lw=1.5, color="black", **kwargs)

        def get_fit_base(pid):
            return ''.join([c for c in pid if not c.isalpha()])

        group_map = {}
        for _, row in self.fitted_peaks.iterrows():
            base_id = get_fit_base(row['peak_id'])
            group_map.setdefault(base_id, []).append(row)

        colormap = colors or plt.cm.get_cmap('tab10')

        for i, (fit_index, rows) in enumerate(group_map.items()):
            rows = pd.DataFrame(rows)
            peak_type = rows.iloc[0]['type']
            bg_type = rows.iloc[0]['bg_type']
            bg_coeffs = rows.iloc[0]['bg_coeffs']
            x_min_fit = rows.iloc[0].get('x_min', None)
            x_max_fit = rows.iloc[0].get('x_max', None)

            plot_min = x_min if x_min is not None else -np.inf
            plot_max = x_max if x_max is not None else np.inf
            fit_min = x_min_fit if x_min_fit is not None else -np.inf
            fit_max = x_max_fit if x_max_fit is not None else np.inf

            if fit_max < plot_min or fit_min > plot_max:
                continue

            fit_mask = (self.x >= max(plot_min, fit_min)) & (self.x <= min(plot_max, fit_max))

            x_fit = self.x[fit_mask]
            if len(x_fit) == 0:
                continue

            color = colormap(i % 10)
            total_fit = np.zeros_like(x_fit, dtype=float)

            for _, peak in rows.iterrows():
                if peak_type == 'gaussian':
                    fit = peak['amp'] * np.exp(-((x_fit - peak['cen']) ** 2) / (2 * peak['wid'] ** 2))
                elif peak_type == 'lorentzian':
                    fit = peak['amp'] / (1 + ((x_fit - peak['cen']) / peak['wid']) ** 2)
                elif peak_type == 'voigt':
                    from scipy.special import wofz
                    sigma = peak['sigma']
                    gamma = peak['gamma']
                    z = ((x_fit - peak['cen']) + 1j * gamma) / (sigma * np.sqrt(2))
                    fit = peak['amp'] * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
                else:
                    raise ValueError(f"Unsupported peak type: {peak_type}")
                
                # Background
                if bg_type == 'constant':
                    bg = np.full_like(x_fit, bg_coeffs[0])
                elif bg_type == 'linear':
                    bg = bg_coeffs[0] + bg_coeffs[1] * x_fit
                elif bg_type == 'cubic':
                    bg = np.polyval(bg_coeffs, x_fit)
                else:
                    raise ValueError(f"Unsupported background type: {bg_type}")

                if show_individual_peaks:
                    ax.plot(x_fit, fit + bg, linestyle='--', color=color, alpha=0.7, label=f"Peak {peak['peak_id']}", **kwargs)

                total_fit += fit

            total_fit += bg

            if show_total_fit:
                ax.plot(x_fit, total_fit, color=color, lw=2.0, label=f"Total Fit {fit_index}", **kwargs)

        if title is not None:
            ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)

        if legend:
            ax.legend()

        if save_path:
            fig.savefig(save_path, dpi=dpi)
            print(f"Overlay plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)
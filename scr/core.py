import numpy as np
from pint import UnitRegistry

ureg = UnitRegistry()

class Spectrum:
    def __init__(self, x, y, x_unit=None, y_unit=None, filename=None):
        """
        Initialize a Spectrum object.
        
        Parameters:
        x : array-like
            The x values of the spectrum (e.g., wavelength, frequency).
        y : array-like
            The y values of the spectrum (e.g., intensity).
        x_unit : str, optional
            The unit of the x values (e.g., 'nm', 'Hz').
        y_unit : str, optional
            The unit of the y values (e.g., 'W/m^2').
        filename : str, optional
            The filename from which the spectrum was loaded.
        """

        self.filename = filename
        
        self._has_units = bool(x_unit or y_unit)

        if self._has_units:
            self.x = np.array(x) * ureg(x_unit)
            self.y = np.array(y) * ureg(y_unit)
        else:
            self.x = np.array(x)
            self.y = np.array(y)
    

    @classmethod
    def from_csv(cls, path, delimiter=',', x_unit=None, y_unit=None):
        """
        Load a Spectrum from a CSV file.
        
        Parameters:
        path : str
            The path to the CSV file.
        delimiter : str, optional
            The delimiter used in the CSV file (default is ',').
        x_unit : str, optional
            The unit of the x values.
        y_unit : str, optional
            The unit of the y values.
        
        Returns:
        Spectrum
            An instance of the Spectrum class.
        """
        data = np.loadtxt(path, delimiter=delimiter)
        x = data[:, 0]
        y = data[:, 1]
        return cls(x, y, x_unit=x_unit, y_unit=y_unit, filename=path)
    

    def convert_x_unit(self, new_unit):
        """
        Convert the x values to a new unit.
        
        Parameters:
        new_unit : str
            The new unit for the x values.
        
        Returns:
        Spectrum
            A new Spectrum instance with converted x values.
        """
        if not self._has_units:
            raise ValueError("Spectrum has no units.")
        
        self.x = self.x.to(new_unit)

    
    def convert_y_unit(self, new_unit):
        """
        Convert the y values to a new unit.
        
        Parameters:
        new_unit : str
            The new unit for the y values.
        
        Returns:
        Spectrum
            A new Spectrum instance with converted y values.
        """
        if not self._has_units:
            raise ValueError("Spectrum has no units.")
        
        self.y = self.y.to(new_unit)


    def convert_units(self, new_x_unit, new_y_unit):
        """
        Convert both x and y values to new units.

        Parameters:
        new_x_unit : str
            The new unit for the x values.
        new_y_unit : str
            The new unit for the y values.
        """
        if not self._has_units:
            raise ValueError("Spectrum has no units.")
        self.x = self.x.to(new_x_unit)
        self.y = self.y.to(new_y_unit)
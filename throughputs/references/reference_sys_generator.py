import numpy as np
import astropy.units as u


class reference_system:

    def __init__(self):
        pass

    def write_to_file(self, filename):
        column1 = self.wavelengths.value
        column2 = self.flux.value

        # print(column1, column2)
        write_object = np.column_stack((column1, column2))
        # print(write_object)
        header = '\n'.join([' This is a file for the {0} reference system.'.format(self.system), ' ', ' ', ' The columns are: wavelengths, flux reference.',
        ' With the following units:', '     {0}         ,   {1}'.format(self.wavelength_unit, self.flux_unit)])
        np.savetxt(filename, write_object, delimiter=' , ', header=header)


class ab_system(reference_system):

    def __init__(self):
        self.base_flux_level = 3631.0*u.Jansky
        self.base_unit = 'Jansky'
        self.system = 'AB'

    def in_cgs_per_angstrom(self, wavelengths=None, wave_units=None):
        if wavelengths is None:
            self.wavelengths = np.linspace(start=1000, stop=20000, num=19001, endpoint=True)*u.Angstrom
        else:
            self.wavelengths = np.asarray(wavelengths)*wave_units
            self.wavelengths.to(u.Angstrom, equivalencies=u.spectral())

        flux = np.empty_like(self.wavelengths.value)

        for i, wave in enumerate(self.wavelengths):
            flux[i] = self.base_flux_level.to(u.erg/(u.s*u.cm*u.cm*u.Angstrom), equivalencies=u.spectral_density(wave.value*u.Angstrom)).value
        self.flux = flux * u.erg/(u.s*u.cm*u.cm*u.Angstrom)

#        self.flux = self.base_flux_level.to(u.erg/(u.s*u.cm*u.cm*u.Angstrom), equivalencies=u.spectral_density(self.wavelengths))
        self.flux_unit = 'ergs/s/cm^2/Angstrom'
        self.wavelength_unit = 'Angstroms'


system = ab_system()
system.in_cgs_per_angstrom()
system.write_to_file('ab.dat')

import sncosmo

class get_sed():
seds_path =

phase, wave, flux = sncosmo.read_griddata_ascii(filename)
source = sncosmo.TimeSeriesSource(phase, wave, flux)
model = sncosmo.Model(source=source)

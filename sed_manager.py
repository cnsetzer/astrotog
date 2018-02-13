import sncosmo


def __init__
    seds_path = "./SEDB"

class sed():


def 
    seds_path = input("Directory ")

    phase, wave, flux = sncosmo.read_griddata_ascii(filename)
    source = sncosmo.TimeSeriesSource(phase, wave, flux)
    model = sncosmo.Model(source=source)

MODULE macronova2py

USE macronova_Pinto_eastman_CNS


CONTAINS

SUBROUTINE Calculate_luminosity(n, MNE_parameters, read_hrate, heating_rates_file, Nt, luminosity)
    IMPLICIT NONE

    INTEGER, INTENT(IN) :: n, Nt
    !f2py Integer, intent(in):: n, Nt
    DOUBLE PRECISION, INTENT(IN)  :: MNE_parameters(n)
    !f2py DOUBLE PRECISION, intent(in), depend(n) :: MNE_parameters
    DOUBLE PRECISION, INTENT(OUT) :: luminosity(Nt+1,4)
    !f2py DOUBLE PRECISION, intent(out) :: luminosity
    LOGICAL, INTENT(IN) :: read_hrate
    !f2py intent(in) :: read_hrate
    CHARACTER*12, INTENT(IN) :: heating_rates_file
    !f2py intent(in) :: heating_rates_file

    CALL Macronova(n, MNE_parameters, read_hrate, heating_rates_file, Nt, luminosity)


END SUBROUTINE Calculate_luminosity


END MODULE

MODULE macronova2py

USE macronova_Pinto_eastman_CNS


CONTAINS

SUBROUTINE Calculate_luminosity(n,MNE_parameters,luminosity)

    IMPLICIT NONE

    DOUBLE PRECISION, INTENT(IN)  :: MNE_parameters(n)
    DOUBLE PRECISION, INTENT(OUT), ALLOCATABLE :: luminosity


    CALL Macronova(MNE_parameters, luminosity)


END SUBROUTINE Calculate_luminosity


END MODULE

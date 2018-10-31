MODULE macronova2py

USE macronova_Pinto_eastman_CNS


CONTAINS

SUBROUTINE calculate_luminosity(N_tran, n, MNE_parameters, read_hrate, heating_rates_file, Nt, luminosity)
    IMPLICIT NONE

    INTEGER, INTENT(IN) :: n, Nt, N_tran
    !f2py Integer, intent(in):: n, Nt, N_tran
    DOUBLE PRECISION, INTENT(IN)  :: MNE_parameters(n,N_tran)
    !f2py DOUBLE PRECISION, intent(in), depend(n,N_tran) :: MNE_parameters
    DOUBLE PRECISION, INTENT(OUT) :: luminosity(Nt+1,4,N_tran)
    !f2py DOUBLE PRECISION, intent(out), depend(Nt,N_tran) :: luminosity
    LOGICAL, INTENT(IN) :: read_hrate
    !f2py intent(in) :: read_hrate
    CHARACTER*255, INTENT(IN) :: heating_rates_file
    !f2py intent(in) :: heating_rates_file

    CALL macronova(N_tran, n, MNE_parameters, read_hrate, heating_rates_file, Nt, luminosity)

END SUBROUTINE calculate_luminosity


END MODULE

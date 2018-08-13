MODULE physics_constants

  IMPLICIT NONE

  DOUBLE PRECISION,PARAMETER:: &
       hplanck=  6.62607004d-27,&! Planck constant [erg*s]
       kB=       1.38064852d-16,&! Boltzmann constant [erg/K]
       parsec=   3.08567758d+18,&! parsec [cm]
       clight=   2.99792458d+10,&! speed of light [cm/s]
       sigma=    5.67051d-5,    &! Stefan-Boltzmann const. [erg/(s*cm2*K4)]
       msol=     1.9891d+33,    &! solar mass [g]
       day_in_s= 8.64d+4,       &! one day [s]
       Robs=     10.0*parsec,   &! distance for abs. mags (10 pc,[cm])
       Ang=      1.0d-8,        &! angstrom [cm]
       Pi=       ASIN(1.0)*2     ! pi (3.14159265358979..)

END MODULE physics_constants

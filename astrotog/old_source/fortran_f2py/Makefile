F90FLAGS = '-cpp -g -O3 -ffpe-trap=overflow,underflow,invalid -Wall -fbacktrace \
		   -fimplicit-none -fdefault-double-8 -fdefault-real-8 -fopenmp'

LAPACK_DIR = /usr/lib64/
FPATH = -L$(LAPACK_DIR) -I$(LAPACK_DIR)
FLIBS = -llapack -lblas

all:
	f2py -c $(FPATH) --f90flags=$(F90FLAGS) $(FLIBS) macronova2py.f90 macronova_Pinto_Eastman_CNS.o module_physics_constants.o -m macronova2py
	cp *.so ../.

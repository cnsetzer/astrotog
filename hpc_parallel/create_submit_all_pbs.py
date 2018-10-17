import os
import sys
import numpy as np
from subprocess import Popen, PIPE

n_seeds = np.int(sys.argv[1])
input_paths = []

for i in range(len(sys.argv) - 2):
    input_paths.append(sys.argv[i+2])

if n_seeds > 1:
    seeds = np.random.randint(low=int(pow(n_seeds, 4)), size=n_seeds).tolist()
else:
    seeds = [12345]

cwd = os.getcwd()
user = cwd.split(sep='/')[1]

for input_path in input_paths:
    inputs = os.listdir(input_path)
    for inp in inputs:
        for seed in seeds:
            job_name = input_path.split(sep='/')[1] + '_' + inp.replace('_inputs.py', '')
            input_fpath = input_path + inp
            input_file = input_fpath.replace('/', '.')
            p = Popen('qsub', stdin=PIPE, stdout=PIPE, close_fds=True, start_new_session=True, shell=False)
            job_string = '''#!/bin/bash --norc
            #PBS -S /bin/bash
            #PBS -V
            #PBS -N {0}
            #PBS -q cores24
            #PBS -l nodes=1:ppn=24,walltime=99:00:00
            #PBS -r n
            #PBS -k oe
            module purge
            module load python
            module load mpi/mvapich/2-2.3b
            source activate astrotog_hpc
            cd {1}

            echo $PBS_O_WORKDIR
            echo Running on host `hostname`
            echo Time is `date`
            echo Directory is `pwd`
            echo PBS job ID is $PBS_JOBID
            echo This jobs runs on the following machines:

            #!Create a machine file
            cat $PBS_NODEFILE | uniq > job_files/machine.file.$PBS_JOBID
            echo PBS_NODEFILE=$PBS_NODEFILE

            export PYTHONPATH=$PYTHONPATH:/home/{2}/.conda/envs/astrotog_hpc/bin/python

            mpirun -genvlist PATH,LD_LIBRARY_PATH,LD_RUN_PATH,PYTHONPATH --machinefile $PBS_NODEFILE python simulation_pipeline.py {3} {4}
            '''.format(job_name, cwd, user, input_file, seed)
            p.stdin.write(job_string)
            p.stdin.close()
            print(p.stdout.read())
            p.wait()

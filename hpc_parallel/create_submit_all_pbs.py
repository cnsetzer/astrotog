import os
import re
import sys
import numpy as np
from subprocess import Popen, PIPE

queue = sys.argv[1]
n_nodes = sys.argv[2]
ppn = sys.argv[3]
env = sys.argv[4]
n_seeds = np.int(sys.argv[5])
script = sys.argv[6]

input_paths = []

for i in range(len(sys.argv) - 7):
    input_paths.append(sys.argv[i + 7])

if n_seeds > 1:
    seeds = np.random.randint(
        4294967295, size=n_seeds
    ).tolist()  # chose seeds within the range [0,max unsigned integer i.e. 4,294,967,295).
else:
    seeds = [123456]

cwd = os.getcwd()
user = cwd.split(sep="/")[1]

for input_path in input_paths:
    inputs = os.listdir(input_path)
    for inp in inputs:
        if re.search("__pycache__", inp):
            continue
        else:
            for i, seed in enumerate(seeds):
                input_fpath = input_path + inp
                input_dumb = input_fpath.replace("/", ".")
                input_file = input_dumb.replace(".py", "")
                job_name = (
                    input_path.split(sep="/")[2]
                    + "_"
                    + inp.replace("_inputs.py", "")
                    + "_s{}".format(i)
                )
                p = Popen(
                    "qsub",
                    stdin=PIPE,
                    stdout=PIPE,
                    close_fds=True,
                    start_new_session=True,
                    shell=False,
                )
                job_string = """#!/bin/bash --norc
                #PBS -S /bin/bash
                #PBS -V
                #PBS -N {0}
                #PBS -q {1}
                #PBS -l nodes={2}:ppn={3},walltime=500:00:00
                #PBS -r n
                #PBS -k oe
                module purge
                module load python
                source activate {4}
                module load mpi/mvapich/2-2.3b
                cd {5}

                echo $PBS_O_WORKDIR
                echo Running on host `hostname`
                echo Time is `date`
                echo Directory is `pwd`
                echo PBS job ID is $PBS_JOBID
                echo This jobs runs on the following machines:

                #!Create a machine file
                cat $PBS_NODEFILE | uniq > job_files/machine.file.$PBS_JOBID
                echo PBS_NODEFILE=$PBS_NODEFILE

                echo The random seed for this job is: {6}

                export PYTHONPATH=$PYTHONPATH:/home/{7}/.conda/envs/{4}/bin/python

                mpirun -genvlist PATH,LD_LIBRARY_PATH,LD_RUN_PATH,PYTHONPATH --machinefile $PBS_NODEFILE -ppn {3} python {8} {9} {6}
                """.format(
                    job_name,
                    queue,
                    n_nodes,
                    ppn,
                    env,
                    cwd,
                    seed,
                    user,
                    script,
                    input_file,
                )
                p.stdin.write(job_string.encode(encoding="utf_8"))
                p.stdin.close()
                out_bytes = p.stdout.read()
                print(out_bytes.decode("utf_8"))
                p.wait()

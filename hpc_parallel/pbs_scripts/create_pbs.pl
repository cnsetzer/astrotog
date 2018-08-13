#!/usr/local/bin/perl

use Cwd;
use File::Basename;

#Use current directory as root
$dir = cwd;

$nargs = scalar(@ARGV);
$code = @ARGV[0];
$queue = @ARGV[1];
$num = @ARGV[2];
$ppn = @ARGV[3];

$path = $dir;
$exe = basename($code);

open(Fout,">./pbs_script");
print Fout <<EMP;
#!/bin/bash -l
#PBS -V
#PBS -N run_$code
#PBS -q $queue
#PBS -l nodes=$num:ppn=$ppn
#PBS -r n
#PBS -j oe
#PBS -k oe
module purge
module load python
module load mpi/mvapich/2-2.3b
source activate astrotog
cd $dir

echo Running on host \`hostname\`
echo Time is \`date\`
echo Directory is \`pwd\`
echo PBS job ID is \$PBS_JOBID
echo This jobs runs on the following machines:
echo \`cat \$PBS_NODEFILE | uniq\`

#! Create a machine file
cat \$PBS_NODEFILE | uniq > ../job_files/machine.file.\$PBS_JOBID
PYTHONPATH=\$PYTHONPATH:~/.conda/envs/astrotog/bin/python

mpirun -genvlist PATH,LD_LIBRARY_PATH,LD_RUN_PATH,PYTHONPATH, --machinefile \$PBS_NODEFILE /share/apps/anaconda/python3.6/bin/python ../sim_scripts/$code

EMP
close(Fout);

@args=("qsub","./pbs_script");
system(@args);
chdir("../");

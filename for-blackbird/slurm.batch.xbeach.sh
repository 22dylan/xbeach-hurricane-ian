#!/bin/bash
#SBATCH -p batch 
#SBATCH --job-name=xbeach
#SBATCH --time=7:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=161

#SBATCH --output=PRE.%j.out
#SBATCH --error=PRE.%j.err

module purge
module load openmpi/4.1.5/gcc-11 
module load netcdf-c/4.7.4/gcc-11        
module load miniforge
module load hdf5/1.14.0/gcc-11    
module load netcdf-fortran/4.5.3/gcc-11

srun -n $SLURM_NTASKS --mpi=pmix_v3 xbeach -parallel > xbeach.log


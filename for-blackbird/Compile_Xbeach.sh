# 0) Clean shell env and load a consistent stack
module purge
# Do NOT also load a generic "gcc" if you're using the gcc-11 OpenMPI stack
module load openmpi/4.1.5/gcc-11
module load hdf5/1.14.0/gcc-11
module load netcdf-c/4.7.4/gcc-11
module load netcdf-fortran/4.5.3/gcc-11
module load miniforge

# 1) Python helper (optional, for XBeach tooling)
pip install --user mako

# 2) Your local env (adds ~/local/bin, libs, and exports NETCDF_* helpers) (linking to Stefano's directory) 
# everyone can read everyone else's directories and files on blackbird  
source /home/snb51/local/env_modules.sh

# 4) Get the source (use your working svn; otherwise specify a target dir)
svn checkout https://svn.oss.deltares.nl/repos/xbeach/trunk 
cd trunk

# 5) Bootstrap (only if autogen.sh exists)
[ -x ./autogen.sh ] && ./autogen.sh

# 6) Configure — pull the right include + link flags from nc/nf-config
#    Also choose an install prefix you own (no sudo needed).
CC=mpicc FC=mpifort \
CPPFLAGS="$NETCDF_CFLAGS" \
FCFLAGS="$NETCDF_FFLAGS" \
LIBS="$NETCDF_FLIBS $NETCDF_CLIBS" \
./configure --with-netcdf --with-mpi --prefix="$HOME/.local"

# 7) Build + install
make -j
make install

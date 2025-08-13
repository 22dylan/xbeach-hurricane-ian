
# note: try running without mpich 
sudo apt install subversion gcc gfortran build-essential libhdf5-dev libnetcdf-dev libnetcdff-dev netcdf-bin openmpi-bin openmpi-doc libopenmpi-dev mpich python-is-python3 python3-mako

# download source code
svn co https://svn.oss.deltares.nl/repos/xbeach/trunk/

# now compiling and installing
cd trunk
./autogen.sh
FCFLAGS="-I/usr/include" ./configure --with-netcdf --with-mpi
make
sudo make install
# ldconfig  # note; may need to run this.


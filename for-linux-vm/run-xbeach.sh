# -to execute model and terminate ssh, run the following command 
# (may need to remove `bash`):
# `screen -S mysession nohup bash run-xbeach.sh`

# - `screen` allows one to terminate ssh connection, but the xbeach processes 
#    will still run.
# - `-S` option creates a screen session called "mysesion"
# - `nohup` writes terminal output to nohup.txt
# - `run-xbeach.sh`  is this file; changes to the run directory below, then
#    executes xbeach in parallel mode. 

cd xbeach-runs-gvm/xbeach/models/run2
mpirun --use-hwthread-cpus xbeach


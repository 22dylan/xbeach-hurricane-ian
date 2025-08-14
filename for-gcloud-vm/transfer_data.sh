GDRIVEPATH="/Users/drs/Google Drive/My Drive/2025-jhu/Reseach/2025-03-27-xbeach-ian"
ZONE="us-west1-a"   # large vm
# ZONE="us-west1-b"   # small vm

# local to vm; model runs
# FROMPATH="$GDRIVEPATH/xbeach/models/run1-5m-bldgs-12hr"
# TOPATH="drs@xb-vm-large:~/xbeach/models"

# ## vm to local; model results
# FROMPATH="drs@xb-vm:~/xbeach/models/run4-bldgs/xboutput.nc"
# TOPATH="$GDRIVEPATH/xbeach/models/run4-bldgs/"

# vm to local; plotting results
FROMPATH="drs@xb-vm-large:~/plotting/plot-output/run1-5m-bldgs-12hr-H.mp4"
# FROMPATH="drs@xb-vm:~/xbeach/plotting/plot-output/gvm-run10-30m-nobldgs-zs.mp4"
TOPATH="/Users/drs/Desktop"

gcloud compute scp --recurse --zone "$ZONE" "$FROMPATH" "$TOPATH"

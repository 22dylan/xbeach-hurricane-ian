GDRIVEPATH="/Users/drs/Google Drive/My Drive/2025-jhu/Reseach/2025-03-27-xbeach-ian"

# # local to vm; model runs
# FROMPATH="$GDRIVEPATH/xbeach/models/gvm-run4-30m-nobldgs"
# TOPATH="drs@xb-vm:~/xbeach/models"

# ## vm to local; model results
# FROMPATH="drs@xb-vm:~/xbeach/models/run4-bldgs/xboutput.nc"
# TOPATH="$GDRIVEPATH/xbeach/models/run4-bldgs/"

# vm to local; plotting results
# FROMPATH="drs@xb-vm:~/xbeach/plotting/plot-output/temp.png"
FROMPATH="drs@xb-vm:~/xbeach/plotting/plot-output/gvm-run5-30m-nobldgs-H.mp4"
TOPATH="/Users/drs/Desktop"

gcloud compute scp --recurse --zone "us-west1-b" "$FROMPATH" "$TOPATH"

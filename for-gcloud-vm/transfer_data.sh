GDRIVEPATH="/Users/drs/Google Drive/My Drive/2025-jhu/Reseach/2025-03-27-xbeach-ian"
ZONE="us-west1-a"   # large vm
# ZONE="us-west1-b"   # small and medium vm's
# ZONE="us-central1-c"    # for plotting vm

# # local to vm; model runs
# FROMPATH="$GDRIVEPATH/xbeach/models/run14-microdomain-1m-bldgs-3hr-tideloc2"
# TOPATH="drs@xb-vm-large:~/xbeach/models/"

# # vm to local; model results
# FROMPATH="drs@xb-vm:~/xbeach/models/frun13-microdomain-1m-bldgs-3hr-tideloc1-tt2/xboutput.nc"
# TOPATH="$GDRIVEPATH/xbeach/models/frun13-microdomain-1m-bldgs-3hr-tideloc1-tt2/"


# vm to local; plotting results
FROMPATH="drs@xb-vm-large:~/plotting/plot-output-video/f129.png"
# # FROMPATH="drs@xb-vm-large:~/plotting/plot-wave-height/run2max.npy"
TOPATH="/Users/drs/Desktop"

gcloud compute scp --recurse --zone "$ZONE" "$FROMPATH" "$TOPATH"

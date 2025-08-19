GDRIVEPATH="/Users/drs/Google Drive/My Drive/2025-jhu/Reseach/2025-03-27-xbeach-ian"
ZONE="us-west1-a"   # large vm
# ZONE="us-west1-b"   # small and medium vm's

# # local to vm; model runs
# FROMPATH="$GDRIVEPATH/xbeach/models/run5-5m-bldgs-6hr-tideloc1-largedomain/"
# TOPATH="drs@xb-vm-large:~/xbeach/models"

# # vm to local; model results
# FROMPATH="drs@xb-vm-large:~/xbeach/models/frun3-30m-bldgs-12hr-tideloc1/"
# TOPATH="$GDRIVEPATH/xbeach/models/"

# vm to local; plotting results
# FROMPATH="drs@xb-vm-large:~/plotting/plot-transect/ytrans5640-t29.png"
# FROMPATH="drs@xb-vm-large:~/plotting/plot-output/temp.png"
FROMPATH="drs@xb-vm-large:~/plotting/plot-output/run5-5m-bldgs-6hr-tideloc1-largedomain-H.mp4"
# FROMPATH="drs@xb-vm-large:~/plotting/plot-output/frun3-30m-bldgs-12hr-tideloc1-H.mp4"
TOPATH="/Users/drs/Desktop"

gcloud compute scp --recurse --zone "$ZONE" "$FROMPATH" "$TOPATH"

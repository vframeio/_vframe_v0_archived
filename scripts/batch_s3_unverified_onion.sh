#!/bin/bash
# ------------------------------------------------------
# Sync keyframe directories to S3
# Usage: ./batch_s3_unverified.sh names_udk.txt
# ------------------------------------------------------
if [ $# -ne 1 ]; 
    then echo "Usage: script.sh names_udk.txt"
    exit 0
fi

# source and destination
filename="$1"
keyframes_src='/data_store_hdd/apps/syrianarchive/media/keyframe/unverified'
keyframes_dst='s3://sa-vframe/v1/media/keyframes/unverified'

# create logfile
#logfile="/media/blue/unverified/${filename}.log"
logfile="_local_onion_s3.log"
echo "# Logfile started at $(date)" > $logfile

# read in file of 3-letter sha256
while read -r line
do
  name="$line"

  echo "$(date) $name"
  #echo "$(date) start $name" >> $logfile

  s3cmd sync $keyframes_src/$name/ $keyframes_dst/$name/
  #echo $cmd
  echo "$(date) done $name" >> $logfile

done < "$filename"

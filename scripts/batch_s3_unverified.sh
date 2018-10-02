#!/bin/bash
# ------------------------------------------------------
# Sync keyframe directories to S3
# Usage: ./batch_s3_unverified.sh names_udk.txt
# ------------------------------------------------------
filename="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"
if [ $# -ne 1 ]; 
    then echo "Usage: $me names_udk.txt"
	  exit 0
fi

# source and destination
filename="$1"
#keyframes_src='/data_store/datasets/syrian_archive/media/keyframes/unverified'
keyframes_src='/media/blue/unverified/data_store/datasets/syrian_archive/media/keyframes/unverified'
keyframes_dst='s3://sa-vframe/v1/media/keyframes/unverified'

# create logfile
logfile="/media/blue/unverified/${filename}.log"
echo "# Logfile started at $(date)" > $logfile

cd /media/blue/unverified

# read in file of 3-letter sha256
while read -r line
do
  name="$line"

  echo "$(date) $name"
  echo "$(date) UNZIP $name" >> $logfile
  unzip -q "${name}.zip"

  cmd="$(date) aws --quiet --endpoint-url https://ams3.digitaloceanspacs.com s3 sync $keyframes_src/$name/ $keyframes_dst/$name/"
  echo $cmd >> $logfile
  aws --quiet --endpoint-url https://ams3.digitaloceanspaces.com s3 sync $keyframes_src/$name/ $keyframes_dst/$name/ --acl public-read

  echo "$(date) REMOVE $name" >> $logfile
  rm -rf "/media/blue/unverified/data_store/"
  echo "$(date) OK $name" >> $logfile
done < "/home/lens/undisclosed/vframe/scripts/$filename"

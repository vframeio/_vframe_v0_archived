#!/bin/bash

# process videos into still frames for testing Simple Image Search
# https://stackoverflow.com/questions/10225403/how-can-i-extract-a-good-quality-jpeg-image-from-an-h264-video-file-with-ffmpeg#10234065

dir_root=/media/adam/ah3tb/data_store/apps/syrian_archive/
dir_videos=$dir_root'snapshot_20171115/barrel-bomb/'
dir_videos_ls=$dir_root'snapshot_20171115/barrel-bomb/*'
dir_stills=$dir_root'still_frames/barrel-bomb/'

mkdir -p $dir_stills

for f in $dir_videos_ls;do 
	fname=$(basename "$f")
	ext="${filename##*.}"
	fname="${fname%.*}"
	vin=$f
	vout=$dir_stills''$fname'/'
	mkdir -p $vout
	# r 1/1 is 1FPS
	ffmpeg -i $vin -qscale:v 2 -r 1/1 $vout''%d.jpg
done

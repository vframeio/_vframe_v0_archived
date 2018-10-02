#!/bin/bash
dir_png='frames'
dir_jpg_base=$dir_png'_jpg'
echo $dir_jpg
mkdir $dir_jpg_base
dir_png_ls="$dir_png/*"
quality=80

for dir_vid in "$dir_png"/*;
do
	
	vid_id=$(basename $dir_vid)
	dir_jpg=$dir_jpg_base'/'$vid_id'/'
	mkdir $dir_jpg
	
    for f_png in "$dir_vid"/*.png;
    do
	    fname_ext=$(basename "$f_png")
		ext="${fname_ext##*.}"
		fname="${fname_ext%.*}"
		f_jpg=$dir_jpg''$fname'.jpg'
    	convert $f_png -format jpg -quality $quality $f_jpg
    done
    echo $dir_jpg
done

#!/bin/bash

dir_src=/data_store/datasets/syrian_archive/media/keyframes/unverified
dir_dst=/data_store/datasets/syrian_archive/media/keyframes/unverified_zips
dir_src_ls=$dir_src'/*'
for d in $dir_src_ls;
do
	# echo $d
	bname=$(basename "$d")
	dname="${bname%.*}"
	fp_zip=$dir_dst/$dname'.zip'
	zip -r $fp_zip $d
	echo $dname' --> '$fp_zip
done
#!/bin/bash

dir_vids=/data_store/datasets/syrian_archive/video_snapshot_20171115/videos/cluster-munition
dir_out=/data_store/datasets/syrian_archive/video_snapshot_20171115/videos/demos
dir_net=/media/adam/hyperdrive/data_store/datasets/syrian_archive/training/cluster_munition_net/clusternet_v2_01a
cfg=$dir_net/clusternet_v2_01a_test.cfg
weights=$dir_net/backup/clusternet_v2_01a_70000.weights
data=$dir_net/clusternet_v2_01a.data


for f in $dir_vids/*;do
	echo $f
	cd /opt/darknet_pjreddie 
	./darknet detector demo $data $cfg $weights $f
	filename=$(basename "$f")
	extension="${filename##*.}"
	filename="${filename%.*}"
	f_out=$dir_out/$filename'.mp4'
	echo $f_out
	mv test_dnn_out.avi $f_out
	#ffmpeg -i test_dnn_out.avi $f_out
done
cd -
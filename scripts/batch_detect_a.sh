#!/bin/bash

batch_size=50000
batch_start=0
batch_max=300000
batch_end=$((batch_start + batch_size))

cd /vframe/vframe
while [ "$batch_end" -lt "$batch_max" ]; do
	batch_start=$((batch_start + batch_size))
	batch_end=$((batch_start + batch_size))
	python cli_vframe.py --unverified source \
		slice $batch_start $batch_end \
		data -t keyframe -t keyframe_status -t mediainfo \
		filter --min medium \
		load -a add --disk ssd --density basic --size medium \
		detect_dk -t submunition \
		load -a rm \
		data -a rm -t keyframe -t keyframe_status -t mediainfo \
		save -o "/data_store_ssd/apps/syrianarchive/metadata/submunition/unverified/index_$batch_start_$batch_end.pkl"
done
cd -
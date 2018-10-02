#!/bin/bash
# ------------------------------------------------------
#
# Sync metadata to vframe ssd storage
# 
# ------------------------------------------------------


# ---------------------------------------------------------------------------
# get input for extension, verified, and direction

usage() { echo "Usage: $0 [-s <pkl|json>] [-s <verified|unverified>] [-d <r2l|l2r>]" 1>&2; exit 1; }

while getopts ":e:v:d:" o; do
    case "${o}" in
        e)
            e=${OPTARG}
            ((e == pkl || e == json)) || usage
            ;;
        v)
            v=${OPTARG}
            ((v == verified || v == uverified)) || usage
            ;;

        d)
            d=${OPTARG}
            ((d == r2l || d == l2r)) || usage
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${e}" ] || [ -z "${v}" ] || [ -z "${d}" ]; then
    usage
fi


# ---------------------------------------------------------------------------
# set vars and iterate metadata directories

metadata_src=/data_store_ssd/apps/syrianarchive/metadata
metadata_dst=/data_store/apps/syrianarchive/metadata

#for t in mediainfo keyframe keyframe_status coco places365 media_record sugarcube openimages feature_vgg16 feature_pt_resnet18 feature_pt_alexnet
for t in feature_resnet18 feature_alexnet
do
    echo '-------------------------------------------------------------------'
    echo "Syncing $t"
    
    if [ "$d" == "l2r" ]; then
        src=$metadata_src/$t/$v/*.$e
        dst=$metadata_dst/$t/$v/
		echo "Sync remote to local: $src --> $dst"
		rsync -avz --progress $src vframe-adam:$dst
	else
        src=$metadata_dst/$t/$v/*.$e
        dst=$metadata_src/$t/$v/
		echo "Sync remote to local: $src --> $dst"
		mkdir -p $metadata_src/$t/$v/
		rsync -avz --progress vframe-adam:$src $dst
	fi
done

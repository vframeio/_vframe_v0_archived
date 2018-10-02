#!/bin/bash
# ------------------------------------------------------
#
# Sync metadata to vframe ssd storage
# 
# ------------------------------------------------------


# ---------------------------------------------------------------------------
# get input for extension, verified, and direction

usage() { echo "Usage: $0 [-d <r2l|l2r>]" 1>&2; exit 1; }

while getopts ":d:" o; do
    case "${o}" in
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

if [ -z "${d}" ]; then
    usage
fi


# ---------------------------------------------------------------------------
# set vars and iterate metadata directories

metadata_src=/data_store_ssd/apps/syrianarchive/metadata
metadata_dst=/data_store/apps/syrianarchive/metadata

# ---------------------------------------------------------------------------
# Add classes file

for t in coco places365 openimages
do
    echo '-------------------------------------------------------------------'
    echo "Syncing classes $t"

    if [ "$d" == "l2r" ]; then
        src=$metadata_src/$t/classes.txt
        dst=$metadata_dst/$t/
        echo "Sync remote to local: $src --> $dst"
        rsync -avz --progress $src vframe-adam:$dst
    else
        src=$metadata_dst/$t/classes.txt
        dst=$metadata_src/$t/
        echo "Sync remote to local: $src --> $dst"
        rsync -avz --progress vframe-adam:$src $dst
    fi
done

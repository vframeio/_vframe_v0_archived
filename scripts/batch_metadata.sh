#!/bin/bash


# metadata (should go on SSD)
metadata_nas='/data_store/datasets/syrian_archive/metadata'
metadata_ssd='/data_store_ssd/datasets/syrian_archive/metadata'
metadata_hdd='/data_store_hdd/datasets/syrian_archive/metadata'
metadata=$metadata_ssd

# media (should on HDD or NAS)
media_nas='/data_store/datasets/syrian_archive/media'
media_ssd='/data_store_ssd/datasets/syrian_archive/media'
media_hdd='/data_store_hdd/datasets/syrian_archive/media'
media=$media_hdd

# ----------------------------------------------------------------
# Generate mappings
# ----------------------------------------------------------------
# convert CSV to JSON|Pickle
# python sugarcube.py map 
# 	-i $metadata_ssd/mappings/sugarcube/20180611.csv \
# 	-o $metadata_ssd/mappings/20180611/verified/index.json \
# 	--status verified
# python sugarcube.py map \
# 	-i $metadata_ssd/mappings/sugarcube/20180611.csv \
# 	-o $metadata_ssd/mappings/20180611/unverified/index.json \
# 	--status unverified
#convert to pickle
# python metadata.py convert \
# -i $metadata_ssd/mappings/20180611/verified/index.json \
# 	-o $metadata_ssd/mappings/20180611/verified/index.pkl
# python metadata.py convert \
# 	-i $metadata_ssd/mappings/20180611/unverified/index.json \
# 	-o $metadata_ssd/mappings/20180611/unverified/index.pkl

# ----------------------------------------------------------------
# Concat mediainfo from tree-files
# ----------------------------------------------------------------
# concatenate json-tree files into single output file
# python metadata.py concat \
# 	-i $metadata_ssd/mappings/20180611/verified/index.pkl \
# 	--metadata $metadata_ssd/mediainfo_tree/ \
# 	-o $metadata_ssd/mediainfo/verified/index.json \
# 	--type mediainfo
# about 35 minutes for 1.236.000 files
# python metadata.py concat \
# 	-i $metadata_ssd/mappings/20180611/unverified/index.pkl \
# 	--metadata $metadata_ssd/mediainfo_tree/ \
# 	-o $metadata_ssd/mediainfo/unverified/index.json \
# 	--type mediainfo



# **************************************************************************
# convert JSON to Pickle
# python metadata.py convert \
# 	-i $metadata_ssd/mediainfo/verified/index.json \
# 	-o $metadata_ssd/mediainfo/verified/index.pkl
# python metadata.py convert \
# 	-i $metadata_ssd/mediainfo/unverified/index.json \
# 	-o $metadata_ssd/mediainfo/unverified/index.pkl


# ----------------------------------------------------------------
# Concat fileattr from tree-files
# ----------------------------------------------------------------
# python metadata.py concat \
# 	-i $metadata_ssd/mappings/20180611/verified/index.pkl \
# 	--metadata $metadata_ssd/mediainfo_tree/ \
# 	-o $metadata_ssd/attributes/verified/index.json \
# 	--type fileattr
# about 35 minutes for 1.236.000 files
# python metadata.py concat \
# 	-i $metadata_ssd/mappings/20180611/unverified/index.pkl \
# 	--metadata $metadata_ssd/mediainfo_tree/ \
# 	-o $metadata_ssd/attributes/unverified/index.json \
# 	--type fileattr
# python metadata.py convert \
# 	-i $metadata_ssd/attributes/verified/index.json \
# 	-o $metadata_ssd/attributes/verified/index.pkl
# python metadata.py convert \
# 	-i $metadata_ssd/attributes/unverified/index.json \
# 	-o $metadata_ssd/attributes/unverified/index.pkl

# ----------------------------------------------------------------
# Concat keyframe from single files
# ----------------------------------------------------------------
# python metadata.py concat \
# 	-i $metadata_ssd/mappings/20180611/verified/index.pkl \
# 	--metadata $metadata_ssd/keyframes_tree/ \
# 	-o $metadata_ssd/keyframes/verified/index.json \
# 	--type keyframe
# python metadata.py concat \
# 	-i $metadata_ssd/mappings/20180611/unverified/index.pkl \
# 	--metadata $metadata_ssd/keyframes_tree/ \
# 	-o $metadata_ssd/keyframes/unverified/index.json \
# 	--type keyframe
# python metadata.py convert \
# 	-i $metadata_ssd/keyframes/verified/index.json \
# 	-o $metadata_ssd/keyframes/verified/index.pkl
# python metadata.py convert \
# 	-i $metadata_ssd/keyframes/unverified/index.json \
# 	-o $metadata_ssd/keyframes/unverified/index.pkl


# ----------------------------------------------------------------
# Append to master file
# ----------------------------------------------------------------
# python metadata.py append -i $metadata_ssd/mappings/20180611/verified/index.pkl \
# 	-m $metadata/attributes/verified/index.pkl \
# 	-m $metadata/keyframes/verified/index.pkl \
# 	-m $metadata/mediainfo/verified/index.pkl \
# 	-o $metadata/master/verified/index.json
# python metadata.py convert \
# 	-i $metadata_ssd/master/verified/index.json \
# 	-o $metadata_ssd/master/verified/index.pkl

# unverified
#python metadata.py append -i $metadata_ssd/mappings/20180611/unverified/index.pkl \
#	-m $metadata/attributes/unverified/index.pkl \
#	-m $metadata/mediainfo/unverified/index.pkl \
#	-m $metadata/keyframes/unverified/index.pkl \
#	-o $metadata/master/unverified/index.json
#python metadata.py convert \
#	-i $metadata_ssd/master/unverified/index.json \
#	-o $metadata_ssd/master/unverified/index.pkl

# ----------------------------------------------------------------
# Extract keyframe files
# ----------------------------------------------------------------
# convert to pickle

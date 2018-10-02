#!/bin/bash
# convert folder of images into video
if [ "$#" -ne 2 ]; then
    #echo "usage: <infolder> <outfilepath> <filename_prefix> <ext> <numdigits> <framerate>"
    echo "usage: <infolder> <outfilepath>"
    exit
fi
# print every line
# set -e, abort if
set -ex
INDIR=$1
OUTFILEPATH=$2
#PREFIX=$3
#PREFIX='frame_'
#PREFIX='2VxY_nwryB8.mp4_frame_'
#EXT=$4
EXT='tiff'
#NUMDIGITS=$5
#NUMDIGITS=8
#FRAMERATE=$6
#FRAMERATE=$7
FRAMERATE=25

# echo 'infolder: '$INDIR
# echo 'outfolder: '$OUTFILEPATH
# echo "extension: " $EXT
# echo 'filename_prefix:'$PREFIX
# echo 'numdigits: '$NUMDIGITS
# echo 'framerate: '$FRAMERATE

#ffmpeg -r 1/5 -i /home/docker/test/tiff/2VxY_nwryB8.mp4_frame_%08d.tiff -c:v libx264 -vf fps=25 -pix_f

#CMD='ffmpeg -i '$INDIR''$PREFIX'%0'$NUMDIGITS'd.'$EXT' -r '$FRAMERATE' -pix_fmt yuv420p '$OUTFILEPATH
#echo $CMD
#eval $CMD

#ffmpeg -framerate $FRAMERATE -pattern_type glob -i '*.tiff' -c:v libx264 -r $FRAMERATE -pix_fmt yuv420p $OUTFILEPATH

#ffmpeg -framerate 1 -pattern_type glob -i '*.jpg' -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4

#cat *.tiff | ffmpeg -f image2pipe -r 25 -i - -vcodec libx264 out.mp4

ffmpeg -r 25 -pattern_type glob -i '0e892a19c1742d62a943bd8ee24fe7106ef5055e25051442aa62117f2aefa6b8.mp4_frame_*.tiff' -c:v libx264 -pix_fmt yuv420p ../out.mp4

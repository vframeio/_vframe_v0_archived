Collection of utility scripts for video, image, and network

-------------------------------

# Video Utilities

## FFMPEG

resize video

`ffmpeg -i input.mov -filter:v scale=720:-1 -c:a copy out.mov`

resize to still frames

`ffmpeg -i input.mov -s 320x180 -qscale:v 2 to/frame_%07d.jpg`

export frames every n seconds

`ffmpeg -i input.flv -vf fps=1 out%d.png`

export frames every x/n seconds

`ffmpeg -i myvideo.avi -vf fps=1/60 img%03d.jpg`

shorten video

`ffmpeg -ss [start] -i in.mp4 -t [duration] -c copy out.mp4`

shorten video example: 5 second clip starting at 5 seconds

`ffmpeg -ss 00:00:05 -i file.mp4 -t 00:00:05 -c copy file_5s.mp4`

render video from still frames:
`ffmpeg -r 1/5 -i frame_%03d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p out.mp4`

using glob:
`ffmpeg -framerate 1 -pattern_type glob -i '*.png' -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4`

or
`ffmpeg -i frame_%03d.png -c:v ffv1 -qscale:v 0 test.avi`

concatenate videos
`ffmpeg -f concat -i inputs.txt -vcodec copy -acodec copy Mux1.mp4`

-------------------------------

# Image Utilities

## Cropping, resizing, convert format

Convert images to maximum width and height using imagemagick

`mogrify in.jpg -resize 800x600`

Resize and crop center

`convert original.jpg -resize 200x200^ -gravity Center -crop 200x200+0+0 +repage cropped.png`

Crop image using imagemagick with W x H x X x Y

`mogrify -crop 640x480+50+100 foo.png`

Convert all files with matching filename to .jpg

- `find . -name "*.gif" -print0 | xargs -0 mogrify -format jpg`

Resize all files with matching name to new size

- `find . -name "*.jpg" -print0 | xargs -0 mogrify -resize 720`

Optimize all images in subdirectories for web / s3:

- `find . -name "*.jpg" -print0 | xargs -0 mogrify -sampling-factor 4:2:0 -strip -quality 85 -interlace JPEG -colorspace RGB`

Verify image is not corrupt

- `identify "./myfolder/*" >log.txt 2>&1`

Resize all images in subdirectories such that smallest dimension is 256

- `find . -name "*.JPEG" | xargs -I {} convert {} -resize "256^>" {}`

Crop animated GIF

- `convert input.gif -coalesce -repage 0x0 -crop WxH+X+Y +repage output.gif`


Not sure what this is:

`-format "%[pixel: u.p{0,0}]" info:`

Batch folder of images into squares, for pix2pix

```
W=64
H=64
INDIR=mydir/orig
OUTDIR=mydir/images

for f in $INDIR/*.jpg;
do
  fn=$(basename "$f")
  convert $f -resize $Wx$H^ -gravity Center -crop $W'x'$H'+0+0' +repage $OUTDIR/$fn
done
```

## Animate GIFs

With ImageMagick

`convert -delay 100 -loop 0 image*.jpg animation.gif`

With ImageMagick, higher quality

`convert -background white -alpha remove -layers OptimizePlus -delay 1 *.png -loop 0 anim.gif`

With [Gifsicle](https://www.lcdf.org/gifsicle/)

`gifsicle --delay=10 --loop *.gif > anim.gif`


-------------------------------


# File I/O Utilities

## Cleaning Directories

Remove all of a specific file type

`find . -name "*.gif" -print0 | xargs -0 rm`

Try removing invalid images

`find . -name \*.png -exec identify -ping {} \; -or -exec echo {} \;`
`find . -name \*.png -exec identify -ping {} \; -or -exec rm -f {} \;`

Rename directory of files incrementally

`j=1;for i in *.png; do mv "$i" ILZMi_MGTDg_q2_"$j".png; let j=j+1;done`

Count all files in subdirectories

`find . -type f | wc -l`

Remove all files from input file

`xargs rm < 1.txt`

Count all files in subdirectories

`find . -maxdepth 1 -mindepth 1 -type d -exec sh -c 'echo "{} : $(find "{}" -type f | wc -l)" file\(s\)' \;`

Filter images with white pixels in upper corner

```
for f in *;do convert $f -format "%[pixel: u.p{0,0}]" info:|xargs echo $f |grep white|cut -d ' ' -f 1|xargs rm -f;done
```

Copy all files in subdirectoreis

`find . -name \*.jpg -exec cp {} ../test \;`

Clean directories after google images python scrape
```
find  -name "*thumb-90x90.jpg" | xargs rm -f
find . -name "*.jpg" | grep -v 'clean' | xargs rm -f
find . -name "*.jpeg" | grep -v 'clean' | xargs rm -f
```

Center crop square images for training

```
convert -define jpeg:size=200x200 original.jpeg  -thumbnail 100x100^ -gravity center -extent 100x100  thumbnail.jpeg
```

Resize all directories to square crops for Caffe
```
find . -name "*.jpg" -print0 | xargs -0 mogrify -resize 256x256^ -gravity Center -crop 256x256+0+0 +repage

```



Replace spaces in filenames

`find . -depth -name "* *" -execdir rename 's/ /_/g' "{}" \;`

Rename all files in lowercase

`find . -depth -exec rename 's/(.*)\/([^\/]*)/$1\/\L$2/' {} \;`

**Delete images less than a specified size. Run without `--delete` to verify list**

`missing cmd here?`

-------------------------------------------

## Example Script for Cleaning ImageNet Downloads

```
for f in *;do mogrify -resize 640x640! $f;done
for f in *;do mogrify -crop 500x500+70+70 $f;done
# filter out images with solid color backgrounds

for f in *;do convert $f -format "%[pixel: u.p{0,0}]" info:|xargs echo $f |grep -E 'white|black|red|blue|green|yellow|purple|none|transparent|WhiteSmoke|gray99|gray98' | cut -d ' ' -f 1|xargs rm -f;done
#
for f in *;do convert $f -format "%[pixel: u.p{499,499}]" info:|xargs echo $f |grep -E 'white|black|red|blue|green|yellow|purple|none|transparent|WhiteSmoke|gray99|gray98' | cut -d ' ' -f 1|xargs rm -f;done
#
for f in *;do convert $f -format "%[pixel: u.p{0,499}]" info:|xargs echo $f |grep -E 'white|black|red|blue|green|yellow|purple|none|transparent|WhiteSmoke|gray99|gray98' | cut -d ' ' -f 1|xargs rm -f;done
#
for f in *;do convert $f -format "%[pixel: u.p{499,0}]" info:|xargs echo $f |grep -E 'white|black|red|blue|green|yellow|purple|none|transparent|WhiteSmoke|gray99|gray98' | cut -d ' ' -f 1|xargs rm -f;done
#
# Remove files that are b&w
for f in *;do convert $f -colorspace HSL -channel g -separate +channel -format "%[fx:mean]" info: | xargs echo $f | grep -v "0.[0-9]*$" | cut -d ' ' -f 1 | xargs rm -f;done
# Remove files that are close to b&w
for f in *;do convert $f -colorspace HSL -channel g -separate +channel -format "%[fx:mean]" info: | xargs echo $f | grep "0.0[0-9]*$" | cut -d ' ' -f 1 | xargs rm -f;done
#
# mk bkup dir at full size (512)
for f in *;do mogrify -resize 256x256! -median 5x5 -quality 100 $f;done
for f in *;do mogrify -median 4x4 -quality 100 $f;done
for f in *;do mogrify -blur 0x6 -quality 100 $f;done
# create AB train,val,test splits
# process A with blur Filter
cd
```


---------------------------------------

# Network Utilities

## Setup Vhosts

```
sudo mkdir -p /var/www/api.dulldream.xyz/public_html
sudo chown -R $USER:$USER /var/www/api.dulldream.xyz/public_html
sudo cp /etc/apache2/sites-available/api.dulldream.conf /etc/apache2/sites-available/api.dulldream.xyz.conf
sudo nano /etc/apache2/sites-available/dulldream.xyz.conf
```

```
<VirtualHost *:80>
    ServerAdmin admin@example.com
    ServerName imagenet.com
    ServerAlias www.imagenet.com
    DocumentRoot /var/www/imagenet.com/public_html
    ErrorLog ${APACHE_LOG_DIR}/error.log
    CustomLog ${APACHE_LOG_DIR}/access.log combined
</VirtualHost>
```
```
sudo a2ensite imagenet.xyz.conf
```

## Local Server

PHP

`cd ~/public_html; php -S localhost:8000`

Python

`python -m SimpleHTTPServer 8000`


Attempt to use modrewrite with built in server
```
 if (file_exists(__DIR__ . '/' . $_SERVER['REQUEST_URI'])) {
     return false; // serve the requested resource as-is.
   } else {
     include_once 'index.php';
   }
```


# System Utilities

configure local timezone

```
export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
sudo dpkg-reconfigure locales
```


-------------------------------------------

## Image Annotation

*BBTag: image annotation tool*

This launches the BBTag annotation app

`/root/BBTag/BBTag.sh`

Process folders labeled with BBTag

This will convert directory of images into cropped/resized images for Caffe training

`Python/convert_bb_tag_xml.py -x input.xml -s source/images -o output/images`
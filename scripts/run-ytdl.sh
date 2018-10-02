#!/bin/bash
# youtube-dl "https://www.youtube.com/watch?v=ILZMi_MGTDg -o %(id)s.%(ext)s -f mp4"
# --prefer-ffmpeg 


youtube-dl "https://www.youtube.com/watch?v=MhwF1z0AnH4" -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best' -o '%(id)s.%(ext)s'
# Convert to still frames
# ffmpeg -ss 00:00:25 -t 00:00:00.00 -i YOURMOVIE.MP4 -r 25.0 YOURIMAGE%4d.jpg
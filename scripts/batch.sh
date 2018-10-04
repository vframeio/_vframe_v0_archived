#!/bin/basdh
# regenerate all the features for resnet
# RAM intensive. limit to one task. running multiple will overload RAM

python  cli_vframe.py --unverified source slice 50000 100000 data -t keyframe -t keyframe_status -t mediainfo filter --min medium images --disk ssd --size medium --density basic features -t resnet18 images -a rm data -a rm -t keyframe -t keyframe_status -t mediainfo save -t feature_resnet18 --suffix 50000_100000.pkl -f 

python  cli_vframe.py --unverified source slice 100000 150000 data -t keyframe -t keyframe_status -t mediainfo filter --min medium images --disk ssd --size medium --density basic features -t resnet18 images -a rm data -a rm -t keyframe -t keyframe_status -t mediainfo save -t feature_resnet18 --suffix 100000_150000.pkl -f

python  cli_vframe.py --unverified source slice 150000 200000 data -t keyframe -t keyframe_status -t mediainfo filter --min medium images --disk ssd --size medium --density basic features -t resnet18 images -a rm data -a rm -t keyframe -t keyframe_status -t mediainfo save -t feature_resnet18 --suffix 150000_200000.pkl -f

python  cli_vframe.py --unverified source slice 200000 250000 data -t keyframe -t keyframe_status -t mediainfo filter --min medium images --disk ssd --size medium --density basic features -t resnet18 images -a rm data -a rm -t keyframe -t keyframe_status -t mediainfo save -t feature_resnet18 --suffix 200000_250000.pkl -f

python  cli_vframe.py --unverified source slice 300000 350000 data -t keyframe -t keyframe_status -t mediainfo filter --min medium images --disk ssd --size medium --density basic features -t resnet18 images -a rm data -a rm -t keyframe -t keyframe_status -t mediainfo save -t feature_resnet18 --suffix 300000_350000.pkl -f

python  cli_vframe.py --unverified source slice 350000 400000 data -t keyframe -t keyframe_status -t mediainfo filter --min medium images --disk ssd --size medium --density basic features -t resnet18 images -a rm data -a rm -t keyframe -t keyframe_status -t mediainfo save -t feature_resnet18 --suffix 350000_400000.pkl -f

python  cli_vframe.py --unverified source slice 400000 450000 data -t keyframe -t keyframe_status -t mediainfo filter --min medium images --disk ssd --size medium --density basic features -t resnet18 images -a rm data -a rm -t keyframe -t keyframe_status -t mediainfo save -t feature_resnet18 --suffix 400000_450000.pkl -f

python  cli_vframe.py --unverified source slice 450000 500000 data -t keyframe -t keyframe_status -t mediainfo filter --min medium images --disk ssd --size medium --density basic features -t resnet18 images -a rm data -a rm -t keyframe -t keyframe_status -t mediainfo save -t feature_resnet18 --suffix 450000_500000.pkl -f

python  cli_vframe.py --unverified source slice 500000 550000 data -t keyframe -t keyframe_status -t mediainfo filter --min medium images --disk ssd --size medium --density basic features -t resnet18 images -a rm data -a rm -t keyframe -t keyframe_status -t mediainfo save -t feature_resnet18 --suffix 500000_550000.pkl -f

python  cli_vframe.py --unverified source slice 550000 600000 data -t keyframe -t keyframe_status -t mediainfo filter --min medium images --disk ssd --size medium --density basic features -t resnet18 images -a rm data -a rm -t keyframe -t keyframe_status -t mediainfo save -t feature_resnet18 --suffix 550000_600000.pkl -f

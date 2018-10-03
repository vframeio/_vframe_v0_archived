#!/bin/bash
# ------------------------------------------------------
#
# Sync metadata to vframe ssd storage
# 
# ------------------------------------------------------


# ---------------------------------------------------------------------------
# get input for extension, verified, and direction

usage() { echo "Usage: $0 [-s <pkl|json>] [-s <verified|unverified>] [-d <r2l|l2r>] -t \"<metadata_type>\"" 1>&2; exit 1; }

# ---------------------------------------------------------------------------
# checks if array contains a value
# @paolo-tedesco https://stackoverflow.com/questions/3685970/check-if-a-bash-array-contains-a-value

function contains() {
    local n=$#
    local value=${!n}
    for ((i=1;i < $#;i++)) {
        if [ "${!i}" == "${value}" ]; then
            echo "y"
            return 0
        fi
    }
    echo "n"
    return 1
}


# ---------------------------------------------------------------------------
# echo colors

RED='\033[0;31m'
NC='\033[0m' # No Color


# ---------------------------------------------------------------------------
# iterated through cli options

while getopts ":e:v:d:t:" o; do
    case "${o}" in
        e)
            e=${OPTARG}
            # ((e == pkl || e == json)) || usage
            ;;
        v)
            v=${OPTARG}
            # ((v == verified || v == uverified)) || usage
            ;;

        d)
            d=${OPTARG}
            # ((d == r2l || d == l2r)) || usage
            ;;
        t)
            t=${OPTARG}
            metadata_types=($t)
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))


# ---------------------------------------------------------------------------
# set default values for args

# force metadata type input
if [ -z "${t}" ]; then
    usage
fi

# set default values for local
if [ -z "${e}" ]; then
    e="pkl"
fi

# set default values for verified
if [ -z "${v}" ]; then
    v="verified"
fi

# set default for verified
if [ -z "${d}" ]; then
    d="l2r"
fi


# ---------------------------------------------------------------------------
# ensure environment variables were set for metadata locations

# TODO change these to env vars
if [ -z "${METADATA_LOCAL}" ]; then
    printf "${RED}Error: ${NC} METADATA_LOCAL was not set\n"
    printf "export METADATA_LOCAL=/local/path/to/your/metadata\n"
    printf "or $ source env/metadata.env\n"
    exit 0
fi
if [ -z "${METADATA_REMOTE}" ]; then
    printf "${RED}Error: ${NC} METADATA_LOCAL was not set\n"
    printf "export METADATA_REMOTE=/remote/path/to/your/metadata\n"
    printf "or $ source env/metadata.env\n"
    exit 0
fi

# ---------------------------------------------------------------------------
# set accepted types of metadata that can be synced
valid_types=("feature_alexnet" "feature_resnet18" "mediainfo" "keyframe" "keyframe_status" "coco" "places365" "media_record" "sugarcube" "openimages" "submunition")


for t in "${metadata_types[@]}"
do
    if [ $(contains "${valid_types[@]}" $t) == "n" ]; then
        printf "${RED}Error: ${NC} \"${t}\" is not a valid metadata type\n"
        continue
    fi

    echo '-------------------------------------------------------------------'
    echo "Syncing $t"
    
    if [ "$d" == "l2r" ]; then
        src=$METADATA_LOCAL/$t/$v/*.$e
        dst=$METADATA_REMOTE/$t/$v/
		echo "local to remote: $src --> $dst"
        echo '-------------------------------------------------------------------'
		rsync -avz --progress $src vframe-adam:$dst
	else
        src=$METADATA_REMOTE/$t/$v/*.$e
        dst=$METADATA_LOCAL/$t/$v/
		echo "remote to local: $src --> $dst"
        echo '-------------------------------------------------------------------'
		mkdir -p $METADATA_LOCAL/$t/$v/
		rsync -avz --progress vframe-adam:$src $dst
	fi
done

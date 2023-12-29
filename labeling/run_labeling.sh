#! /bin/sh

YYYY=${1}
MM=${2}
DD=${3}
HH=${4}

anaDateTime=${YYYY}${MM}${DD}${HH}

# define current directory
currDir=`pwd`

# remove any ERA5_uv_*.grib ir ERA5_uv_*.idx that already exist in directory
rm -f ERA5_uv_*.grib ERA5_uv_*.idx

# download ERA5 GRIB files for anaDateTime observation-window (performed on head node, fails in batch submission)
./run_download_ERA5.sh ${anaDateTime}  

# submit labeling with sub_label.sh

./sub_label.sh ${YYYY} ${MM} ${DD} ${HH}

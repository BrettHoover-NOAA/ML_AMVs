#! /bin/sh

YYYY=${1}
MM=${2}
DD=${3}
HH=${4}

anaDateTime=${YYYY}${MM}${DD}${HH}
era5TPrefix="e5.oper.an.ml.0_5_0_0_0_t.regn320sc."
era5SPPrefix="e5.oper.an.ml.128_134_sp.regn320sc."

# define current directory
currDir=`pwd`

# remove any e5.oper.an.ml.*.nc that already exist in directory
#rm -f e5.oper.an.ml.*.nc

# download ERA5 GRIB files for anaDateTime observation-window (performed on head node, fails in batch submission)
#./run_download_ERA5.sh ${anaDateTime}  

# submit CNN tiling with sub_CNN_tiles.sh for tile 00 (-3 to -2 hrs)
anaWindowBeg="-3"
anaWindowEnd="-2"
anaWindowIndex="00"
#./sub_CNN_tiles_test.sh ${YYYY} ${MM} ${DD} ${HH} ${anaWindowBeg} ${anaWindowEnd} ${anaWindowIndex} ${era5TPrefix} ${era5SPPrefix}
./sub_CNN_tiles.sh ${YYYY} ${MM} ${DD} ${HH} ${anaWindowBeg} ${anaWindowEnd} ${anaWindowIndex} ${era5TPrefix} ${era5SPPrefix}

# submit CNN tiling with sub_CNN_tiles.sh for tile 01 (-2 to -1 hrs)
anaWindowBeg="-2"
anaWindowEnd="-1"
anaWindowIndex="01"
#./sub_CNN_tiles_test.sh ${YYYY} ${MM} ${DD} ${HH} ${anaWindowBeg} ${anaWindowEnd} ${anaWindowIndex} ${era5TPrefix} ${era5SPPrefix}
./sub_CNN_tiles.sh ${YYYY} ${MM} ${DD} ${HH} ${anaWindowBeg} ${anaWindowEnd} ${anaWindowIndex} ${era5TPrefix} ${era5SPPrefix}

# submit CNN tiling with sub_CNN_tiles.sh for tile 02 (-1 to -0 hrs)
anaWindowBeg="-1"
anaWindowEnd="-0"
anaWindowIndex="02"
#./sub_CNN_tiles_test.sh ${YYYY} ${MM} ${DD} ${HH} ${anaWindowBeg} ${anaWindowEnd} ${anaWindowIndex} ${era5TPrefix} ${era5SPPrefix}
./sub_CNN_tiles.sh ${YYYY} ${MM} ${DD} ${HH} ${anaWindowBeg} ${anaWindowEnd} ${anaWindowIndex} ${era5TPrefix} ${era5SPPrefix}

# submit CNN tiling with sub_CNN_tiles.sh for tile 03 (0 to 1 hrs)
anaWindowBeg="0"
anaWindowEnd="1"
anaWindowIndex="03"
#./sub_CNN_tiles_test.sh ${YYYY} ${MM} ${DD} ${HH} ${anaWindowBeg} ${anaWindowEnd} ${anaWindowIndex} ${era5TPrefix} ${era5SPPrefix}
./sub_CNN_tiles.sh ${YYYY} ${MM} ${DD} ${HH} ${anaWindowBeg} ${anaWindowEnd} ${anaWindowIndex} ${era5TPrefix} ${era5SPPrefix}

# submit CNN tiling with sub_CNN_tiles.sh for tile 04 (1 to 2 hrs)
anaWindowBeg="1"
anaWindowEnd="2"
anaWindowIndex="04"
#./sub_CNN_tiles_test.sh ${YYYY} ${MM} ${DD} ${HH} ${anaWindowBeg} ${anaWindowEnd} ${anaWindowIndex} ${era5TPrefix} ${era5SPPrefix}
./sub_CNN_tiles.sh ${YYYY} ${MM} ${DD} ${HH} ${anaWindowBeg} ${anaWindowEnd} ${anaWindowIndex} ${era5TPrefix} ${era5SPPrefix}

# submit CNN tiling with sub_CNN_tiles.sh for tile 05 (2 to 3 hrs)
anaWindowBeg="2"
anaWindowEnd="3"
anaWindowIndex="05"
#./sub_CNN_tiles_test.sh ${YYYY} ${MM} ${DD} ${HH} ${anaWindowBeg} ${anaWindowEnd} ${anaWindowIndex} ${era5TPrefix} ${era5SPPrefix}
./sub_CNN_tiles.sh ${YYYY} ${MM} ${DD} ${HH} ${anaWindowBeg} ${anaWindowEnd} ${anaWindowIndex} ${era5TPrefix} ${era5SPPrefix}

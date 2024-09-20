#! /bin/bash
# datetime settings
YYYY=${1}
MM=${2}
DD=${3}
HH=${4}
anaWindowBeg=${5}
anaWindowEnd=${6}
anaWindowIndex=${7}
era5TPrefix=${8}
era5SPPrefix=${9}

anaDateTime=${YYYY}${MM}${DD}${HH}
tileExt="10"
obDataDir="/scratch1/NCEPDEV/stmp4/Brett.Hoover/ML_AMVs/clustering/testing/${anaDateTime}"
inFile="gdas.t${HH}z.satwnd.tm00.bufr_d_${anaDateTime}_superobs.nc"
outFile="gdas.t${HH}z.satwnd.tm00.bufr_d_${anaDateTime}_superobs_CNN_tile_${anaWindowIndex}.nc"

runDir=`pwd`
obsDir=/scratch1/NCEPDEV/stmp4/Brett.Hoover/ML_AMVs/clustering/testing/${anaDateTime}

# 1a) label observations with ERA5 temperature tiles (superobs)
jName_label_obs=CNN_tile${anaWindowIndex}.SO.${anaDateTime}
jid_label_obs=$(sbatch --parsable -A da-cpu -N 1 -t 0:15:00 --job-name=${jName_label_obs} run_generate_CNN_tiles_ERA5.sh ${anaDateTime} ${anaWindowBeg} ${anaWindowEnd} ${tileExt} ${obDataDir} ${inFile} ${runDir} ${era5TPrefix} ${era5SPPrefix} ${outFile})


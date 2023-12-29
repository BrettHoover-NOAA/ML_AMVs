#! /bin/bash
# datetime settings
YYYY=${1}
MM=${2}
DD=${3}
HH=${4}
anaDateTime=${YYYY}${MM}${DD}${HH}

runDir=`pwd`
obsDir=/scratch1/NCEPDEV/stmp4/Brett.Hoover/ML_AMVs/clustering/testing/${anaDateTime}

# 1a) label observations with ERA5 winds (superobs)
obsFileName=gdas.t${HH}z.satwnd.tm00.bufr_d_${anaDateTime}_superobs.nc
labelsFileName=gdas.t${HH}z.satwnd.tm00.bufr_d_${anaDateTime}_superobs_labels_ERA5.nc
jName_label_obs=label_obs.SO.${anaDateTime}
jid_label_obs=$(sbatch --parsable -A da-cpu -N 1 -t 0:15:00 --job-name=${jName_label_obs} run_label_superobs.sh ${anaDateTime} ${obsDir} ${obsFileName} ${runDir} ${labelsFileName})

# 1b) label observations with ERA5 winds (AMVs)
obsFileName=gdas.t${HH}z.satwnd.tm00.bufr_d_${anaDateTime}_reconciled.nc
labelsFileName=gdas.t${HH}z.satwnd.tm00.bufr_d_${anaDateTime}_labels_ERA5.nc
jName_label_obs=label_obs.AMV.${anaDateTime}
jid_label_obs=$(sbatch --parsable -A da-cpu -N 1 -t 0:15:00 --job-name=${jName_label_obs} run_label_superobs.sh ${anaDateTime} ${obsDir} ${obsFileName} ${runDir} ${labelsFileName})


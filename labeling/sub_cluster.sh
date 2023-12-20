#! /bin/bash
# datetime settings
YYYY=2023
MM=01
DD=01
HH=00
anaDateTime=${YYYY}${MM}${DD}${HH}

runDir=`pwd`
obsDir=/scratch1/NCEPDEV/stmp4/Brett.Hoover/ML_AMVs/clustering/testing/${anaDateTime}

# 1) download ERA5 hourly data on pressure surfaces
jName_download_ERA5=download_ERA5.${anaDateTime}
jid_download_ERA5=$(sbatch --parsable -A da-cpu -N 1 -t 0:15:00 --job-name=${jName_download_ERA5} run_download_ERA5.sh ${anaDateTime})
# 2a) label observations with ERA5 winds (superobs)
obsFileName=gdas.t${HH}z.satwnd.tm00.bufr_d_${anaDateTime}_superobs.nc
labelsFileName=gdas.t${HH}z.satwnd.tm00.bufr_d_${anaDateTime}_superobs_labels_ERA5.nc
jName_label_obs=label_obs.${anaDateTime}
jid_label_obs=$(sbatch --parseable -A da-cpu -N 1 -t 0:15:00 --dependency=afterany:${jid_download_ERA5} --job-name=${jName_label_obs} ${anaDateTime} ${obsDir} ${obsFileName} ${runDir} ${labelsFileName})
# 2b) label observations with ERA5 winds (AMVs)
obsFileName=gdas.t${HH}z.satwnd.tm00.bufr_d_${anaDateTime}.nc
labelsFileName=gdas.t${HH}z.satwnd.tm00.bufr_d_${anaDateTime}_labels_ERA5.nc
jName_label_obs=label_obs.${anaDateTime}
jid_label_obs=$(sbatch --parseable -A da-cpu -N 1 -t 0:15:00 --dependency=afterany:${jid_download_ERA5} --job-name=${jName_label_obs} ${anaDateTime} ${obsDir} ${obsFileName} ${runDir} ${labelsFileName})


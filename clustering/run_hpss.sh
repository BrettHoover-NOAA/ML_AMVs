#! /bin/sh

YYYY=2023

for MM in 05 06 07
do
  sbatch hpss_AMVs.sh ${YYYY}${MM}
  sbatch hpss_SuperObs.sh ${YYYY}${MM}
  sbatch hpss_Labels.sh ${YYYY}${MM}
done

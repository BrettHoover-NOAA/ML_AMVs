#! /bin/sh

YYYY=2022
for MM in 03
do
    sbatch hpss_CNNdat.sh ${YYYY}${MM}
done


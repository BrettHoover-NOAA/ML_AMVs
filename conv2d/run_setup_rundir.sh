#! /bin/sh -f

YYYY=2022
MM=03

for DD in 03
do
  for HH in 00 06 12 18
  do
    ./setup_rundir.sh ${YYYY} ${MM} ${DD} ${HH}
  done
done


#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --time=0:30:00
#SBATCH -A da-cpu
#SBATCH --partition=service
#SBATCH -J hpss-datastore

YYYYMM=${1}    # year-month of data to store (e.g., 202301)
currDir=`pwd`

module load hpss

set -x

hpssdir=${hpssdir:-/NCEPDEV/emc-da/2year/Brett.Hoover/HERA/scratch/ML_AMVs/datastore/AMVs}    # Location of your file in HPSS
tarfile=${tarfile:-${YYYYMM}.tar}                                                             # Name of the tar file in HPSS

#
#   Check if the tarfile index exists.  If it does, assume that
#   the data for the corresponding directory has already been
#   tarred and saved.
#
#
# hsi "ls -l ${hpssdir}/${tarfile}.idx"
# tar_file_exists=$?
# if [ $tar_file_exists -eq 0 ]
#  then
#   echo "File $tarfile already saved."
#  exit
# fi

#   htar is used to create the archive, -P creates
#   the directory path if it does not already exist,
#   and an index file is also made.
#
 htar -P -cvf ${hpssdir}/$tarfile ${YYYYMM}*/gdas.t*z.satwnd.tm00.bufr_d_*_reconciled.nc
 err=$?
 if [ $err -ne 0 ]
   then
     echo "File $tarfile was not successfully created."
   exit 3
 fi

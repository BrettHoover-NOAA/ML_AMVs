#! /bin/sh

DMPDIR=/scratch1/NCEPDEV/global/glopara/dump
CDUMP=gdas
runDir=${1}
YYYY=${2}
MM=${3}
DD=${4}
HH=${5}

srcFile=${DMPDIR}/${CDUMP}.${YYYY}${MM}${DD}/${HH}/atmos/${CDUMP}.t${HH}z.satwnd.tm00.bufr_d
locFile=${runDir}/${CDUMP}.t${HH}z.satwnd.tm00.bufr_d_${YYYY}${MM}${DD}${HH}
if [ -e "${srcFile}" ]; then
    cp ${srcFile} ${locFile}
else
    echo "SOURCE FILE ${srcFile} NOT FOUND"
fi


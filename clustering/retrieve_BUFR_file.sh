#! /bin/sh

DMPDIR=/scratch1/NCEPDEV/global/glopara/dump
CDUMP=gdas
YYYY=2023
MM=04
DD=03
HH=12

srcFile=${DMPDIR}/${CDUMP}.${YYYY}${MM}${DD}/${HH}/atmos/${CDUMP}.t${HH}z.satwnd.tm00.bufr_d
locFile=./${CDUMP}.t${HH}z.satwnd.tm00.bufr_d_${YYYY}${MM}${DD}${HH}
if [ -e "${srcFile}" ]; then
    cp ${srcFile} ${locFile}
else
    echo "SOURCE FILE ${srcFile} NOT FOUND"
fi


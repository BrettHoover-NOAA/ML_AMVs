#! /bin/sh

CDUMP=gdas
runDir=${1}
anaDateTime=${2}
anaHH=${3}

echo "" > rm_list.txt

# if a combined CNN netCDF file was produced, delete any tiles
srcFile=${runDir}/${CDUMP}.t${anaHH}z.satwnd.tm00.bufr_d_${anaDateTime}_superobs_CNN_T.nc
if [ -e "${srcFile}" ]; then
    ls -1 ${runDir}/${CDUMP}.t${anaHH}z.satwnd.tm00.bufr_d_${anaDateTime}_superobs_Tile*.nc >> rm_list.txt
fi
# bundle all slurm-*.out files into a tar-file and remove the text files
tar -cvf ${runDir}/slurm.out ${runDir}/slurm-*.out
ls -1 ${runDir}/slurm-*.out >> rm_list.txt


# remove all files scheduled for removal, followed by rm_list.txt
if [ -e "rm_list.txt" ]; then
    nFiles=`wc -l rm_list.txt | awk '{print $1}'`
    n=0
    while [ ${n} -lt ${nFiles} ]
    do
        ((n++))
        srcFile=`awk "NR==${n}" rm_list.txt`
        if [ -e "${srcFile}" ]; then
            rm "${srcFile}"
        fi
    done
    rm "rm_list.txt"
fi


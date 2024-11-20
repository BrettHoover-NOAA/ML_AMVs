#! /bin/bash

dataDir=${1}
anaDateTime=${2}
anaHH=${3}

runTime=0:20:00

# create job-name
jName_convert=convert_${anaDateTime}

# submit job
jid_convert=$(sbatch --parsable -A da-cpu -N 1 -t ${runTime} --job-name=${jName_convert} run_convert_CNN_file.sh ${dataDir} ${anaDateTime} ${anaHH})

#! /bin/sh
# source bashrc
source /home/Brett.Hoover/.bashrc
### NO ADDITIONAL CHANGES BELOW THIS LINE ###
#############################################
# Load conda envrionment:
# Identify conda base environment: conda info command produces the following:
# base environment : <ENVIRONMENT>  (writable)
# The awk command will select the 4th element space-delimted (<ENVIRONMENT>)
condaBaseEnv=`conda info | grep -i 'base environment' | awk '{print $4}'`
echo "conda base environment: ${condaBaseEnv}"
# Source conda.sh from base environment:
source ${condaBaseEnv}/etc/profile.d/conda.sh
# Activate conda environement:
conda activate ML-labeling
conda env list
#############################################
#
# Execution
#
dataDir=${1}
searchString=${2}
outFile=${3}
numTiles=${4}
anaDateTime=${5}
anaHH=${6}

python combine_CNN_tiles.py "${dataDir}" "${searchString}" "${outFile}" "${numTiles}"

# after completion of python program, run cleanup_rundir.sh to tidy workspace
# IF outFile was produced, and gzip the outFile
if [ -e "${outFile}" ]; then
    ./cleanup_rundir.sh ${dataDir} ${anaDateTime} ${anaHH}
    gzip ${outFile}
fi

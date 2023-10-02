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
conda activate ML-clustering
conda env list
#############################################
#
# Execution
#
runDir="${1}"
anaDateTime=${2}
anaHH=${3}
numTiles=${4}
outputNetcdfFileName="gdas.t${anaHH}z.satwnd.tm00.bufr_d_${anaDateTime}_reconciled.nc"
searchStr="gdas.t${anaHH}z.satwnd.tm00.bufr_d_${anaDateTime}_Tile_"

python reconcile_tiles.py ${runDir} ${searchStr} ${outputNetcdfFileName} ${numTiles}

# after completion of python program, run cleanup_rundir.sh to tidy workspace
# IF outputNetcdfFileName was produced
if [ -e "${outputNetcdfFileName}" ]; then
    ./cleanup_rundir.sh ${runDir} ${anaDateTime} ${anaHH}
fi

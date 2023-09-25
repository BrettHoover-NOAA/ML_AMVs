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
runDir=${1}
anaDateTime=${2}
anaHH=${3}
nPreBins=${4}
minTilePre=${5}
maxTilePre=${6}
haloPre=${7}
nTimBins=${8}
minTileTim=${9}
maxTileTim=${10}
haloTim=${11}
optBins=${12}
netcdfFileName=gdas.t${anaHH}z.satwnd.tm00.bufr_d_${anaDateTime}.nc
python create_tile_yaml.py ${runDir} ${netcdfFileName} ${nPreBins} ${minTilePre} ${maxTilePre} ${haloPre} ${nTimBins} ${minTileTim} ${maxTileTim} ${haloTim} ${optBins} > tiles.yaml

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
nPreBins=${1}
minTilePre=${2}
maxTilePre=${3}
haloPre=${4}
nTimBins=${5}
minTileTim=${6}
maxTileTim=${7}
haloTim=${8}
python create_tile_yaml.py ${nPreBins} ${minTilePre} ${maxTilePre} ${haloPre} ${nTimBins} ${minTileTim} ${maxTileTim} ${haloTim} > tiles.yaml

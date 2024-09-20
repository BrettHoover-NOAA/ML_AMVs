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
anaDateTime=${1}
anaWindowBeg=${2}
anaWindowEnd=${3}
tileExt=${4}
obDataDir=${5}
inFile=${6}
eraDataDir=${7}
era5TPrefix=${8}
era5SPPrefix=${9}
outFile=${10}

python generate_CNN_tiles_ERA5.py "${anaDateTime}" "${anaWindowBeg}" "${anaWindowEnd}" "${tileExt}" "${obDataDir}" "${inFile}" "${eraDataDir}" "${era5TPrefix}" "${era5SPPrefix}" "${outFile}"

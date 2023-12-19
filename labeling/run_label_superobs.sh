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
obDataDir=${2}
inputNetcdfFile=${3}
era5DataDir=${4}
outputNetcdfFile=${5}

python label_superobs_ERA5.py ${anaDateTime} ${obDataDir} ${inputNetcdfFile} ${era5DataDir} ${outputNetcdfFile}


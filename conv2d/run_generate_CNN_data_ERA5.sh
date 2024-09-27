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
dataDir=${2}
subGridExt=${3}
obFile=${4}
era5TPrefix=${5}
era5SPPrefix=${6}
yamlFile=${7}
tileName=${8}
outFile=${9}

python generate_CNN_data_ERA5.py "${anaDateTime}" "${dataDir}" "${subGridExt}" "${obFile}" "${era5TPrefix}" "${era5SPPrefix}" "${yamlFile}" "${tileName}" "${outFile}"

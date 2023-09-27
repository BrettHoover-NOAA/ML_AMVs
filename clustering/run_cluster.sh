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
tileName=${4}
threshDist=${5}
threshPres=${6}
threshTime=${7}
threshUwnd=${8}
threshVwnd=${9}

netcdfFileName=gdas.t${anaHH}z.satwnd.tm00.bufr_d_${anaDateTime}.nc
yamlFile=tiles.yaml

python assign_AMV_clusters.py ${anaDateTime} ${runDir} ${netcdfFileName} ${yamlFile} ${tileName} ${threshDist} ${threshPres} ${threshTime} ${threshUwnd} ${threshVwnd}


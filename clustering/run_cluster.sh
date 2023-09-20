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
anaDateTime=${1}
anaHH=${2}
dataDir=/scratch1/NCEPDEV/stmp4/Brett.Hoover/ML_AMVs/clustering
netcdfFileName=gdas.t${anaHH}z.satwnd.tm00.bufr_d_${anaDateTime}.nc
yamlFile=tiles.yaml
tileName=${3}

python assign_AMV_clusters.py ${anaDateTime} ${dataDir} ${netcdfFileName} ${yamlFile} ${tileName}


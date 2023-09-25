#! /bin/sh
# source bashrc
source /home/Brett.Hoover/.bashrc
### NO ADDITIONAL CHANGES BELOW THIS LINE ###
#############################################
# Load modules
#module purge
#module use modulefiles
#module load BUFR/hera
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
conda activate python_bufr
conda env list
#############################################
#
# Execution
#
runDir=${1}
anaDateTime=${2}
anaHH=${3}
bufrFileName=gdas.t${anaHH}z.satwnd.tm00.bufr_d_${anaDateTime}
netcdfFileName=${bufrFileName}.nc


python process_AMVs_from_BUFR.py ${anaDateTime} ${runDir} ${bufrFileName} ${netcdfFileName}


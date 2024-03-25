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
conda activate ML-learning
conda env list
#############################################
#
# Execution
#
trainDir=${1}
validDir=${2}
epoch=${3}
annealing=${4}
runDir=${5}
modelName=${6}

#python train_SupErrNet_epoch.py ${trainDir} ${validDir} ${epoch} ${annealing} ${runDir} ${modelName}
python train_SupErrNet5.py ${trainDir} ${validDir} ${epoch} ${annealing} ${runDir} ${modelName}

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
nPreBins=6
nTimBins=5
preMin=12500.0
preMax=107500.0
timMin=-3.0
timMax=3.0
preHalo=5000.0
timHalo=0.5
python create_tile_yaml.py ${nPreBins} ${nTimBins} ${preMin} ${preMax} ${timMin} ${timMax} ${preHalo} ${timHalo} > tiles.yaml

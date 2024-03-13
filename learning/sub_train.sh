#! /bin/bash
runDir=`pwd`
trainDir=${runDir}/training
validDir=${runDir}/validation
nEpochs=10

jName_train=train_ep_${nEpochs}
jid_train=$(sbatch --parsable -A da-cpu -N 1 -t 2:00:00 --job-name=${jName_train} run_train_SupErrNet.sh ${trainDir} ${validDir} ${nEpochs} ${runDir} SupErrNet_H6P9)


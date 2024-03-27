#! /bin/bash
runDir=`pwd`
trainDir=${runDir}/training
validDir=${runDir}/validation
nEpochs=10
annealing=0.25
modelName=SupErrNet6_H2P32

runTime=2:00:00

# first epoch: 0
# no prior model is loaded before training
# no dependencies
let epoch=0
# generate jobName based on modelName and epoch
jName_train=train_${modelName}_ep_${epoch}
# submit job and retain jobID
jid_train=$(sbatch --parsable -A da-cpu -N 1 -t ${runTime} --job-name=${jName_train} run_train_SupErrNet.sh ${trainDir} ${validDir} ${epoch} ${annealing} ${runDir} ${modelName})

# loop through epochs starting from one
let epoch=1
let jid_prior=${jid_train}
while [ ${epoch} -lt ${nEpochs} ]
do
    # generate jobID based on modelName and epoch
    jName_train=train_${modelName}_ep_${epoch}
    # submit job with jid_prior as a dependency
    jid_train=$(sbatch --parsable -A da-cpu -N 1 -t ${runTime} --dependency=afterany:${jid_prior} --job-name=${jName_train} run_train_SupErrNet.sh ${trainDir} ${validDir} ${epoch} ${annealing} ${runDir} ${modelName})
    # increment epoch
    ((epoch++))
    # assign current jobID as jid_prior
    jid_prior=${jid_train}
done


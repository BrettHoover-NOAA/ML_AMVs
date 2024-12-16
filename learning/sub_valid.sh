#! /bin/bash
runDir=`pwd`
validDir=${runDir}/validation
statsFile=/scratch1/NCEPDEV/da/Brett.Hoover/ML_AMVs/superob_stats.nc
nEpochs=10
modelName=SupErrNet_v1.0

runTime=2:00:00

####################################################################################
# loop through epochs starting from zero
let epoch=8
while [ ${epoch} -lt ${nEpochs} ]
do
    # generate jobID based on modelName and epoch
    jName_valid=valid_${modelName}_ep_${epoch}
    # submit job with jid_prior as a dependency
    jid_valid=$(sbatch --parsable -A da-cpu -N 1 -t ${runTime} --job-name=${jName_valid} run_validate_SupErrNet.sh ${validDir} ${statsFile} ${epoch} ${runDir} ${modelName})
    # increment epoch
    ((epoch++))
done


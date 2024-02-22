#! /bin/bash
# datetime settings
YYYY=${1}
MM=${2}
DD=${3}
HH=${4}
# define analysis datetime
anaDateTime=${YYYY}${MM}${DD}${HH}
# directory settings
repoDir=/scratch1/NCEPDEV/stmp4/Brett.Hoover/ML_AMVs/labeling
runDir=/scratch1/NCEPDEV/stmp4/Brett.Hoover/ML_AMVs/labeling/testing/${anaDateTime}
currDir=`pwd`

# begin log
echo "" > ${currDir}/log.${anaDateTime}

# set up runDir
#   Check if runDir already exists, if not, create it
if [ ! -d "${runDir}" ]; then
    echo "creating runDir:${runDir}" >> log.${anaDateTime}
    mkdir -p "${runDir}"  # keep double-quotes around these variables in mk/rm commands in case variable was passed empty
else
    echo "runDir:${runDir} already exists"
fi

# copy required scripts and code from repoDir to runDir
#    if repoDir does not exist, report error and exit
if [ ! -d "${repoDir}" ]; then
    echo "ERROR: repoDir:${repoDir} does not exist, exiting"
    exit 1000
else
#   if any srcFile does not exist, report error and exit, otherwise copy from repoDir to runDir
    # sub_label.tmpl: handles all setup and SLURM execution of tasks
    srcFile=${repoDir}/sub_label.sh
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2001
    fi
    # run_download_ERA5.sh: downloads ERA5 GRIB files from Copernicus to run-directory (runs on head-node)
    srcFile=${repoDir}/run_download_ERA5.sh
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2002
    fi
    # download_ERA5.py: performs tasks for run_download_ERA5.sh
    srcFile=${repoDir}/download_ERA5.py
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2003
    fi
    # download_ERA5_CISL.py: performs tasks for run_download_ERA5.sh (NCAR RDA-CISL source)
    srcFile=${repoDir}/download_ERA5_CISL.py
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2004
    fi
    # run_label_superobs.sh: computes u- and v-component labels for input winds
    srcFile=${repoDir}/run_label_superobs.sh
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2005
    fi
    # label_superobs_ERA5.py: performs tasks for run_label_superobs.sh using ERA5 data
    srcFile=${repoDir}/label_superobs_ERA5.py
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2006
    fi
    # label_superobs_ERA5_CISL.py: performs tasks for run_label_superobs.sh using ERA5 data (NCAR RDA-CISL source)
    srcFile=${repoDir}/label_superobs_ERA5_CISL.py
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2007
    fi
    # sub_label.sh: submits labeling tasks to batch queue
    srcFile=${repoDir}/sub_label.sh
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2008
    fi
    # run_labeling.sh: handles running run-scripts in proper order
    srcFile=${repoDir}/run_labeling.sh
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2009
    fi
fi
# move to runDir
cd ${runDir}
# run run_labeling.sh and pass datetime components
./run_labeling.sh ${YYYY} ${MM} ${DD} ${HH}
# return to prior directory
cd -

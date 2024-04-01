#! /bin/bash
# datetime settings
YYYY=${1}
MM=${2}
DD=${3}
HH=${4}
# define analysis datetime
anaDateTime=${YYYY}${MM}${DD}${HH}
# clustering settings
threshDist=100.0       # default setting: 100.0 (km)
threshPres=2500.0      # default setting: 5000.0 (Pa)
threshTime=0.5        # default setting: 0.5 (frac. hrs)
threshUwnd=5.0         # default setting: 5.0 (m/s)
threshVwnd=5.0         # default setting: 5.0 (m/s)
# tile settings
nPreBins=4
minTilePre=10000.0     # pad minimum pressure to avoid errors when computing optimal tiles
maxTilePre=110000.0    # pad maximum pressure to avoid errors when computing optimal tiles
haloPre=${threshPres}  # pressure halo is set to threshPres
nTimBins=4
minTileTim=-3.1        # pad minimum time to avoid errors when computing optimal tiles
maxTileTim=3.1         # pad maximum time to avoid errors when computing optimal tiles
haloTim=${threshTime}  # time halo is set to threshTime
optBins=True
memPerTile=8g          # default setting: 2300M
# directory settings
repoDir=/scratch1/NCEPDEV/stmp4/Brett.Hoover/ML_AMVs/clustering
cpythonExec=/scratch1/NCEPDEV/da/Brett.Hoover/SATWNDBUFR/python_bufr/ioda-bundle/build/lib64/python3.10/pyiodaconv/bufr.cpython-310-x86_64-linux-gnu.so
runDir=/scratch1/NCEPDEV/stmp4/Brett.Hoover/ML_AMVs/clustering/testing/${anaDateTime}
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
    # sub_cluster.tmpl: template-file for sub_cluster.tmpl, which handles all setup and SLURM execution of tasks
    srcFile=${repoDir}/sub_cluster.tmpl
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2001
    fi
    # retrieve_BUFR_file.sh: copies BUFR file from DMPDIR to run-directory (runs on head-node)
    srcFile=${repoDir}/retrieve_BUFR_file.sh
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2002
    fi
    # exec_process_BUFR.sh: sets up module environment for run_process_BUFR.sh
    srcFile=${repoDir}/exec_process_BUFR.sh
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2003
    fi
    # run_process_BUFR.sh: BUFR ingest to netCDF out, computes (u,v) components of wind and time in fractional hours
    srcFile=${repoDir}/run_process_BUFR.sh
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2004
    fi
    # process_satwnds_dependencies.py: functions for process_AMVs_from_BUFR.py
    srcFile=${repoDir}/process_satwnds_dependencies.py
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2005
    fi
    # cpythonExec: BUFR cpython executable from iota-converters for BUFR query
    srcFile=${cpythonExec}
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2006
    fi
    # moduleFile: modulefile for loading module environment for BUFR query
    srcFile=${repoDir}/modulefiles/BUFR/hera.lua
    if [ -e "${srcFile}" ]; then
        # place in its own modulefiles/BUFR subdirectory
        mkdir -p "${runDir}/modulefiles/BUFR"
        cp "${srcFile}" ${runDir}/modulefiles/BUFR/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2007
    fi
    # process_AMVs_from_BUFR.py: performs computational tasks for run_process_BUFR.sh
    srcFile=${repoDir}/process_AMVs_from_BUFR.py
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2008
    fi
    # exec_process_BUFR.sh: sets up module environment for run_process_BUFR.sh
    srcFile=${repoDir}/exec_process_BUFR.sh
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2009
    fi
    # run_create_tile_yaml.sh: generates tiles.yaml to drive clustering in distributed/parallel tile+halo approach
    srcFile=${repoDir}/run_create_tile_yaml.sh
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2010
    fi
    # create_tile_yaml.py: performs computational tasks of run_create_tile_yaml.sh
    srcFile=${repoDir}/create_tile_yaml.py
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2011
    fi
    # run_cluster.sh: performs clustering in distributed/parallel tile+halo approach
    srcFile=${repoDir}/run_cluster.sh
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2012
    fi
    # assign_AMV_clusters.py: performs computational tasks of run_cluster.sh
    srcFile=${repoDir}/assign_AMV_clusters.py
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2013
    fi
    # run_combine_clusters.sh: reconciles clustering from tile+halo approach, final output netCDF of clustering
    srcFile=${repoDir}/run_combine_clusters.sh
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2014
    fi
    # reconcile_tiles.py: performs computational tasks of run_combine_clusters.sh
    srcFile=${repoDir}/reconcile_tiles.py
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2015
    fi
    # run_compute_superobs.sh: performs averaging/rescaling/metadata-production from clustered AMVs, outputs to netCDF
    srcFile=${repoDir}/run_compute_superobs.sh
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2016
    fi
    # compute_superobs.py: performs computational tasks of run_compute_superobs.sh
    srcFile=${repoDir}/compute_superobs.py
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2017
    fi
    # cleanup_rundir.sh: removes intermediate files and collects log files at end of run_combine_clusters.sh
    srcFile=${repoDir}/cleanup_rundir.sh
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2018
    fi
fi
# Generate sub_cluster.sh from sub_cluster.tmpl
#
# Fill in template place-holders with options defined at top
cp ${runDir}/sub_cluster.tmpl ${runDir}/sub_cluster.sh
sed -i "s/>>YYYY<</${YYYY}/g" ${runDir}/sub_cluster.sh
sed -i "s/>>MM<</${MM}/g" ${runDir}/sub_cluster.sh
sed -i "s/>>DD<</${DD}/g" ${runDir}/sub_cluster.sh
sed -i "s/>>HH<</${HH}/g" ${runDir}/sub_cluster.sh
sed -i "s/>>THRESHDIST<</${threshDist}/g" ${runDir}/sub_cluster.sh
sed -i "s/>>THRESHPRES<</${threshPres}/g" ${runDir}/sub_cluster.sh
sed -i "s/>>THRESHTIME<</${threshTime}/g" ${runDir}/sub_cluster.sh
sed -i "s/>>THRESHUWND<</${threshUwnd}/g" ${runDir}/sub_cluster.sh
sed -i "s/>>THRESHVWND<</${threshVwnd}/g" ${runDir}/sub_cluster.sh
sed -i "s/>>NPREBINS<</${nPreBins}/g" ${runDir}/sub_cluster.sh
sed -i "s/>>MINTILEPRE<</${minTilePre}/g" ${runDir}/sub_cluster.sh
sed -i "s/>>MAXTILEPRE<</${maxTilePre}/g" ${runDir}/sub_cluster.sh
sed -i "s/>>HALOPRE<</${haloPre}/g" ${runDir}/sub_cluster.sh
sed -i "s/>>NTIMBINS<</${nTimBins}/g" ${runDir}/sub_cluster.sh
sed -i "s/>>MINTILETIM<</${minTileTim}/g" ${runDir}/sub_cluster.sh
sed -i "s/>>MAXTILETIM<</${maxTileTim}/g" ${runDir}/sub_cluster.sh
sed -i "s/>>HALOTIM<</${haloTim}/g" ${runDir}/sub_cluster.sh
sed -i "s/>>OPTBINS<</${optBins}/g" ${runDir}/sub_cluster.sh
sed -i "s/>>MEMPERTILE<</${memPerTile}/g" ${runDir}/sub_cluster.sh
# chmod ${runDir}/sub_cluster.sh
chmod 700 ${runDir}/sub_cluster.sh
# move to runDir and run sub_cluster.sh, passing full-path to log-file
cd ${runDir}
./sub_cluster.sh ${currDir}/log.${anaDateTime}
cd -

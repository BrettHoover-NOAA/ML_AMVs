#! /bin/bash
# datetime settings
YYYY=${1}
MM=${2}
DD=${3}
HH=${4}
# define analysis datetime
anaDateTime=${YYYY}${MM}${DD}${HH}
# tile settings
nPreBins=4
minTilePre=10000.0     # pad minimum pressure to avoid errors when computing optimal tiles
maxTilePre=110000.0    # pad maximum pressure to avoid errors when computing optimal tiles
haloPre=0.0            # pressure halo is set to zero (no need for halos for this tiling)
nTimBins=4
minTileTim=-3.1        # pad minimum time to avoid errors when computing optimal tiles
maxTileTim=3.1         # pad maximum time to avoid errors when computing optimal tiles
haloTim=0.0            # time halo is set to zero (no need for halos for this tiling)
optBins=True
memPerTile=8g          # default setting: 2300M
# cnn data settings
subGridExt=10
era5TPrefix=e5.oper.an.ml.0_5_0_0_0_t.regn320sc.
era5SPPrefix=e5.oper.an.ml.128_134_sp.regn320sc.
# directory settings
repoDir=/scratch1/NCEPDEV/stmp4/Brett.Hoover/ML_AMVs/conv2d
runDir=/scratch1/NCEPDEV/stmp4/Brett.Hoover/ML_AMVs/conv2d/testing/${anaDateTime}
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
    # sub_CNN_data.tmpl: template-file for sub_CNN_data.sh
    srcFile=${repoDir}/sub_CNN_data.tmpl
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2000
    fi
    # run_download_ERA5.sh: downloads ERA5 GRIB files from Copernicus to run-directory (runs on head-node)
    srcFile=${repoDir}/run_download_ERA5.sh
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2001
    fi
    # download_ERA5_CISL.py: performs tasks for run_download_ERA5.sh (NCAR RDA-CISL source)
    srcFile=${repoDir}/download_ERA5_CISL.py
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2002
    fi
    # run_create_tile_yaml.sh: creates YAML defining subset-tiles for multi-processing
    srcFile=${repoDir}/run_create_tile_yaml.sh
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2003
    fi
    # create_tile_yaml.py: performs functions of run_create_tile_yaml.sh
    srcFile=${repoDir}/create_tile_yaml.py
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2004
    fi
    # run_generate_CNN_data_ERA5.sh: computes CNN temperature data for input winds
    srcFile=${repoDir}/run_generate_CNN_data_ERA5.sh
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2005
    fi
    # generate_CNN_data_ERA5.py: performs tasks for run_generate_CNN_data_ERA5.sh using ERA5 data
    srcFile=${repoDir}/generate_CNN_data_ERA5.py
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2006
    fi
    # run_combine_CNN_tiles.sh: combines CNN temperature data from pressure-time tiles into a single
    #                           and properly ob-ordered file
    srcFile=${repoDir}/run_combine_CNN_tiles.sh
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2007
    fi
    # combine_CNN_tiles.py: performs tasks for run_combine_CNN_tiles.sh
    srcFile=${repoDir}/combine_CNN_tiles.py
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2008
    fi
    # cleanup_rundir.sh: performs cleanup tasks for run_combine_CNN_tiles.sh after completion
    srcFile=${repoDir}/cleanup_rundir.sh
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2009
    fi
    # sub_CNN_data.sh: submits CNN data production tasks to batch queue
    srcFile=${repoDir}/sub_CNN_data.sh
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2010
    fi
    # run_CNN_data.sh: handles running run-scripts in proper order
    srcFile=${repoDir}/run_CNN_data.sh
    if [ -e "${srcFile}" ]; then
        cp "${srcFile}" ${runDir}/.
    else
        echo "ERROR source file:${srcFile} does not exist, exiting"
        exit 2011
    fi
fi

# Generate sub_CNN_data.sh from sub_CNN_data.tmpl
#
# Fill in template place-holders with options defined at top
cp ${runDir}/sub_CNN_data.tmpl ${runDir}/sub_CNN_data.sh
sed -i "s/>>YYYY<</${YYYY}/g" ${runDir}/sub_CNN_data.sh
sed -i "s/>>MM<</${MM}/g" ${runDir}/sub_CNN_data.sh
sed -i "s/>>DD<</${DD}/g" ${runDir}/sub_CNN_data.sh
sed -i "s/>>HH<</${HH}/g" ${runDir}/sub_CNN_data.sh
sed -i "s/>>NPREBINS<</${nPreBins}/g" ${runDir}/sub_CNN_data.sh
sed -i "s/>>MINTILEPRE<</${minTilePre}/g" ${runDir}/sub_CNN_data.sh
sed -i "s/>>MAXTILEPRE<</${maxTilePre}/g" ${runDir}/sub_CNN_data.sh
sed -i "s/>>HALOPRE<</${haloPre}/g" ${runDir}/sub_CNN_data.sh
sed -i "s/>>NTIMBINS<</${nTimBins}/g" ${runDir}/sub_CNN_data.sh
sed -i "s/>>MINTILETIM<</${minTileTim}/g" ${runDir}/sub_CNN_data.sh
sed -i "s/>>MAXTILETIM<</${maxTileTim}/g" ${runDir}/sub_CNN_data.sh
sed -i "s/>>HALOTIM<</${haloTim}/g" ${runDir}/sub_CNN_data.sh
sed -i "s/>>OPTBINS<</${optBins}/g" ${runDir}/sub_CNN_data.sh
sed -i "s/>>MEMPERTILE<</${memPerTile}/g" ${runDir}/sub_CNN_data.sh
sed -i "s/>>SUBGRIDEXT<</${subGridExt}/g" ${runDir}/sub_CNN_data.sh
sed -i "s/>>ERA5TPREFIX<</${era5TPrefix}/g" ${runDir}/sub_CNN_data.sh
sed -i "s/>>ERA5SPPREFIX<</${era5SPPrefix}/g" ${runDir}/sub_CNN_data.sh
# chmod ${runDir}/sub_CNN_data.sh
chmod 700 ${runDir}/sub_CNN_data.sh
# move to runDir and run sub_CNN_data.sh, passing full-path to log-file
cd ${runDir}
./sub_CNN_data.sh ${currDir}/log.${anaDateTime}
cd -


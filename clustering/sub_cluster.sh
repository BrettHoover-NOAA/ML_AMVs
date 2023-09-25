#! /bin/bash
# datetime settings
YYYY=2023
MM=04
DD=04
HH=06
anaDateTime=${YYYY}${MM}${DD}${HH}
# tile settings
nPreBins=6
minTilePre=10000.0
maxTilePre=110000.0
haloPre=5000.0
nTimBins=5
minTileTim=-3.0
maxTileTim=3.0
haloTim=0.5
optBins=True
# directory settings
#repoDir=/scratch1/NCEPDEV/stmp4/Brett.Hoover/ML_AMVs/clustering
#runDir=${repoDir}/testing/${anaDateTime}

runDir=`pwd`

# 0) retrieve BUFR file
echo "retrieving BUFR file" >> log.${anaDateTime}
./retrieve_BUFR_file.sh ${runDir} ${YYYY} ${MM} ${DD} ${HH}
# 1) process BUFR file to netCDF:
#   1a. extract ob and metadata information from selected tanks
#   1b. assign ob-type and pre-QC check flag
#   1c. compute ob-time as fractional hours relative to analysis-time
#   1d. compute u- and v-components of wind from speed and direction
#   1e. write to netCDF file
jName_process_BUFR=process_bufr.${anaDateTime}
jid_process_BUFR=$(sbatch --parsable -A da-cpu -N 1 -t 0:30:00 --job-name=${jName_process_BUFR} exec_process_BUFR.sh ${runDir} ${anaDateTime} ${HH})
# 2) define tiles
jName_generate_tiles=generate_tiles.${anaDateTime}
jid_generate_tiles=$(sbatch --parsable -A da-cpu -N 1 -t 0:10:00 --dependency=afterany:${jid_process_BUFR} --job-name=${jName_generate_tiles} run_create_tile_yaml.sh ${runDir} ${anaDateTime} ${HH} ${nPreBins} ${minTilePre} ${maxTilePre} ${haloPre} ${nTimBins} ${minTileTim} ${maxTileTim} ${haloTim} ${optBins})
# 3) cluster on tiles
#   3a. loop through all Tile_xxx tile-names
for num in 001 002 003 004 005 006 007 008 009 010\
           011 012 013 014 015 016 017 018 019 020\
           021 022 023 024 025 026 027 028 029 030
do
#   3b. compute cluster-ID (cid) and tile-ID (tid) for each observation in tile+halo
#   3c. write tile+halo to netCDF file
    jName_cluster_tiles=cluster_tiles.${anaDateTime}
    sbatch -A da-cpu -N 1 -t 0:30:00 --dependency=afterany:${jid_generate_tiles} --job-name=${jName_cluster_tiles} run_cluster.sh ${runDir} ${anaDateTime} ${HH} Tile_${num}
done
# 4) reconcile tiles
#   4a. collect all tile+halo data from individual tile netCDF files
#   4b. reconcile any duplicate observations to assign a single cid and tid value
#   4c. give each cluster from each tile a unique cid
#   4d. output to netCDF
jName_combine_clusters=${jName_cluster_tiles}  # this is identical to job-name for clustering tiles for use of singleton dependency
sbatch -A da-cpu -N 1 -t 0:10:00 --dependency=singleton --job-name=${jName_combine_clusters} run_combine_clusters.sh ${runDir} ${anaDateTime} ${HH}


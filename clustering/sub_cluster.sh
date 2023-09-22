#! /bin/bash
# datetime settings
YYYY=2023
MM=04
DD=03
HH=18
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

jid_process_BUFR=$(sbatch --parsable -A da-cpu -N 1 -t 0:30:00 --job-name=process_bufr exec_process_BUFR.sh ${anaDateTime} ${HH})
#
jid_generate_tiles=$(sbatch --parsable -A da-cpu -N 1 -t 0:10:00 --dependency=afterany:$jid_process_BUFR --job-name=generate_tiles run_create_tile_yaml.sh ${anaDateTime} ${HH} ${nPreBins} ${minTilePre} ${maxTilePre} ${haloPre} ${nTimBins} ${minTileTim} ${maxTileTim} ${haloTim} ${optBins})
#
for num in 001 002 003 004 005 006 007 008 009 010\
           011 012 013 014 015 016 017 018 019 020\
           021 022 023 024 025 026 027 028 029 030
do
    sbatch -A da-cpu -N 1 -t 0:30:00 --dependency=afterany:$jid_generate_tiles --job-name=cluster_tiles run_cluster.sh ${anaDateTime} ${HH} Tile_${num}
done
#
sbatch -A da-cpu -N 1 -t 0:10:00 --dependency=singleton --job-name=cluster_tiles run_combine_clusters.sh ${anaDateTime} ${HH}

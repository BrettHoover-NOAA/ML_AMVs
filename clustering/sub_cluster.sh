#! /bin/bash

YYYY=2023
MM=04
DD=03
HH=06
anaDateTime=${YYYY}${MM}${DD}${HH}

jid_process_BUFR=$(sbatch --parsable -A da-cpu -N 1 -t 0:30:00 --job-name=process_bufr exec_process_BUFR.sh ${anaDateTime} ${HH})


for num in 001 002 003 004 005 006 007 008 009 010\
           011 012 013 014 015 016 017 018 019 020\
           021 022 023 024 025 026 027 028 029 030
do
    sbatch -A da-cpu -N 1 -t 0:30:00 --dependency=afterany:$jid_process_BUFR --job-name=cluster_tiles run_cluster.sh ${anaDateTime} ${HH} Tile_${num}
done

sbatch -A da-cpu -N 1 -t 0:10:00 --dependency=singleton --job-name=cluster_tiles run_combine_clusters.sh ${anaDateTime} ${HH}

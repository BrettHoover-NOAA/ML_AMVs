#! /bin/sh

for num in 001 002 003 004 005 006 007 008 009 010\
           011 012 013 014 015 016 017 018 019 020\
           021 022 023 024 025 026 027 028 029 030
do
    sbatch -A da-cpu -N 1 -t 0:30:00 run_cluster.sh Tile_${num}
done

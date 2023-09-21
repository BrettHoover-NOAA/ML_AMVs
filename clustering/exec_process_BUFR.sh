#! /bin/tcsh

module purge
module use modulefiles
module load BUFR/hera
module list

set anaDateTime = ${1}
set anaHH = ${2}
./run_process_BUFR.sh ${anaDateTime} ${anaHH}

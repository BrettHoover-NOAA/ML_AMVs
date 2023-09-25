#! /bin/tcsh

module purge
module use modulefiles
module load BUFR/hera
module list

set runDir = ${1}
set anaDateTime = ${2}
set anaHH = ${3}
./run_process_BUFR.sh ${runDir} ${anaDateTime} ${anaHH}

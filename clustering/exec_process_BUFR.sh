#! /bin/tcsh

module purge
module use modulefiles
module load BUFR/hera
module list

set anaDateTime = 2023040306
set anaHH = 06
./run_process_BUFR.sh ${anaDateTime} ${anaHH}

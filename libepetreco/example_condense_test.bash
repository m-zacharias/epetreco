#!/bin/bash
# 
# File:   example_condense_test.bash
# Author: malte
#
# Created on 26.11.2014, 16:52:33
#



###############################################
###
### PARAMETER RANGE
###
###############################################

LIST_BLOCKS="1 7 16 64"
#LIST_BLOCKS="1"
LIST_TPB="1 7 16 64"
#LIST_TPB="1 64"
LIST_SEED="1234"
LIST_THRESHOLD="0.5"
LIST_NTEST="1000"
LIST_BOATSIZE="5 16"
#LIST_BOATSIZE="5"
LIST_TRUCKSIZE="64"
#LIST_TRUCKSIZE="1"

ID_SET=0
for blocks in      $LIST_BLOCKS; do
  for tpb in         $LIST_TPB; do
    for seed in        $LIST_SEED; do
      for threshold in   $LIST_THRESHOLD; do
        for ntest in       $LIST_NTEST; do
          for boatsize in    $LIST_BOATSIZE; do
            for trucksize in   $LIST_TRUCKSIZE; do
              paramsets[$ID_SET]="$blocks $tpb $seed $threshold $ntest $boatsize $trucksize"
              ID_SET=$(echo $ID_SET '+ 1' | bc)
done; done; done; done; done; done; done



###############################################
###
### INDIVIDUALLY DESIGNED PARAMETERSETS
###
###############################################

#paramsets[$ID_SET]="1 1 123 0.5 1000 16 64"
#ID_SET=$(echo $ID_SET '+ 1' | bc)
#
#paramsets[$ID_SET]="1 64 1234 0.5 1000 5 64"
#ID_SET=$(echo $ID_SET '+ 1' | bc)
#
#paramsets[$ID_SET]="7 7 1234 0.5 1000 5 64"
#ID_SET=$(echo $ID_SET '+ 1' | bc)



###############################################
###
### EXECUTE
###
###############################################

MAXID_SET=$(echo $ID_SET '- 1' | bc)
for i in $(seq 0 $MAXID_SET); do
  IFS=" " read -a paramset <<< ${paramsets[$i]}
  blocks=${paramset[0]}
  tpb=${paramset[1]}
  seed=${paramset[2]}
  threshold=${paramset[3]}
  ntest=${paramset[4]}
  boatsize=${paramset[5]}
  trucksize=${paramset[6]}
  rm example_condense_main.out
  PREFIX=$blocks"_"$tpb"_"$seed"_"$threshold"_"$ntest"_"$boatsize"_"$trucksize
  PROGRAM=$PREFIX".out"
  OUT_FN=$PREFIX".dat"
  LOG_FN=$PREFIX".log"
  make CUCFLAGS+=-DNBLOCKS=$blocks \
       CUCFLAGS+=-DTPB=$tpb \
       CUCFLAGS+=-DSEED=$seed \
       CUCFLAGS+=-DTHRESHOLD=$threshold \
       CUCFLAGS+=-DNTEST=$ntest \
       CUCFLAGS+=-DBOATSIZE=$boatsize \
       CUCFLAGS+=-DTRUCKSIZE=$trucksize \
       example_condense_main.out
  mv example_condense_main.out $PROGRAM
  cuda-memcheck $PROGRAM b $OUT_FN > $LOG_FN
#
#  ./example_condense_main.out b $blocks"_"$tpb"_"$seed"_"$threshold"_"$ntest"_"$boatsize"_"$trucksize".dat"
#
#  ./example_condense_main.out b $blocks"_"$tpb"_"$seed"_"$threshold"_"$ntest"_"$boatsize"_"$trucksize".dat" \
#  > $blocks"_"$tpb"_"$seed"_"$threshold"_"$ntest"_"$boatsize"_"$trucksize".log"
done

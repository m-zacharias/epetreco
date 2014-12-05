#!/bin/bash
# 
# File:   example_condense_test.bash
# Author: malte
#
# Created on 26.11.2014, 16:52:33
#

#LIST_BLOCKS="1 2 3 4 5 6 7 8 16 32 64"
LIST_BLOCKS="2"
#LIST_TPB="1 2 3 4 5 6 7 8 16 32 64"
LIST_TPB="1"
LIST_SEED="1234"
LIST_THRESHOLD="0.5"
LIST_NTEST="500"
LIST_T_TEST_SIZE="7"
LIST_TRUCKSIZE="64"

for blocks in      $LIST_BLOCKS; do
for tpb in         $LIST_TPB; do
for seed in        $LIST_SEED; do
for threshold in   $LIST_THRESHOLD; do
for ntest in       $LIST_NTEST; do
for t_test_size in $LIST_T_TEST_SIZE; do
for trucksize in   $LIST_TRUCKSIZE; do
  rm example_condense_main.out
  make CUCFLAGS+=-DNBLOCKS=$blocks \
       CUCFLAGS+=-DTPB=$tpb \
       CUCFLAGS+=-DSEED=$seed \
       CUCFLAGS+=-DTHRESHOLD=$threshold \
       CUCFLAGS+=-DNTEST=$ntest \
       CUCFLAGS+=-DT_TEST_SIZE=$t_test_size \
       CUCFLAGS+=-DTRUCKSIZE=$trucksize \
       example_condense_main.out
  ./example_condense_main.out $blocks"_"$tpb"_"$seed"_"$threshold"_"$ntest"_"$t_test_size"_"$trucksize".txt"
done; done; done; done; done; done; done
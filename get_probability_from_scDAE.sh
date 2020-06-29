#!/bin/bash

for i in {1..5}
do
 train_X="training_10_celltypes_X"
 train_Y="training_10_celltypes_Y"
 test_X="testing_11_celltypes_X"
 test_Y="testing_11_celltypes_Y"
 num=$i

 python scDAE_sm.py $train_X $train_Y $test_X $test_Y $num
done

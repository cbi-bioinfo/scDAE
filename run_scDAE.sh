#!/bin/bash

train_X="training_dataset.csv"
train_Y="cell_type_label_for_training.csv"
test_X="testing_dataset.csv"
test_Y="cell_type_label_for_testing.csv"

python scDAE.py $train_X $train_Y $test_X $test_Y

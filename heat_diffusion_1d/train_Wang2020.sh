#!/bin/bash

export PYTHONPATH='./'

for run_ndx in {1..30}
do
	python train.py \
		--outputDirectory="./output_train_Wang2020_30kepochs/run_${run_ndx}" \
		--randomSeed=$run_ndx \
		--initialProfile=./heated_segments.csv \
		--architecture=Wang2020_2_6_64_1 \
		--duration=10.0 \
		--alpha=0.0001 \
		--scheduleFilepath=./schedule_30k.csv \
		--numberOfBoundaryPoints=256 \
		--numberOfDiffEquResPoints=256
done
#!/bin/bash
#export GRB_LICENSE_FILE=/home/gurobi.lic

max_iter=(20)
learning_rate=(10.0)
decay=(0.8)
tolerance=(0.03)

for mi in "${max_iter[@]}"
do
    for lr in "${learning_rate[@]}"
    do
      for dc in "${decay[@]}"
      do
        for tl in "${tolerance[@]}"
        do
	        python joint_model.py \
	        --max_iter ${mi} \
	        --lr ${lr} \
	        --decay ${dc} \
	        --tolerance ${tl} \
	        --constraints ../data/tbd/top_7.csv \
	        --test_split "test"
        done
      done
    done
done

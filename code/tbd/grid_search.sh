#!/bin/bash
#export GRB_LICENSE_FILE=/home/gurobi.lic

max_iter=(20)
learning_rate=(1.0 2.0 5.0 10.0)
decay=(0.7 0.8 0.9)
tolerance=(0.05 0.03 0.02)

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
	        --constraints ../data/tbd/top_6.csv \
	        --test_split "dev"
        done
      done
    done
done

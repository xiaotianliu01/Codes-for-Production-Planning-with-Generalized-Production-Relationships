#!/bin/bash
python train.py --data_pth='./data/MSOM-06-038-R2.xls' --env_name='SMLP' --experiment_name='Fixed_Production_Lead_Time' --instance_id='01' --seed=1 --reward_lambda=0.0 --episode_length 400 --no_improvement_episodes 20

python train.py --data_pth='./data/MSOM-06-038-R2.xls' --env_name='SMLP' --experiment_name='Clearing_Function_Frac' --instance_id='01' --seed=1 --reward_lambda=0.0 --episode_length 100 --no_improvement_episodes 50

python train.py --data_pth='./data/MSOM-06-038-R2.xls' --env_name='SMLP' --experiment_name='Clearing_Function_Exp' --instance_id='01' --seed=1 --reward_lambda=0.0 --episode_length 100 --no_improvement_episodes 50

python train.py --data_pth='./data/MSOM-06-038-R2.xls' --env_name='SMLP' --experiment_name='Clearing_Function_Poly' --instance_id='01' --seed=1 --reward_lambda=0.0 --episode_length 100 --no_improvement_episodes 50

python train.py --data_pth='./data/MSOM-06-038-R2.xls' --env_name='SMLP_SCHED' --experiment_name='Flow_Shop' --instance_id='01' --seed=1 --reward_lambda=0.0 --episode_length 100 --no_improvement_episodes 20
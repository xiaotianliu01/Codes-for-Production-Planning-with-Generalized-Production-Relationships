#!/bin/bash
python eval.py --env_name='SMLP' --experiment_name='Fixed_Production_Lead_Time' --instance_id='01' --seed=1 --model_dir './results/SMLP/Fixed_Production_Lead_Time/ins-01_lam-0.2_seed-1/models/' --n_eval_rollout_threads 100 --episode_length 400

python eval.py --env_name='SMLP' --experiment_name='Clearing_Function_Frac' --instance_id='01' --seed=1 --model_dir './results/SMLP/Clearing_Function_Frac/ins-01_lam-0.2_seed-1/models/' --n_eval_rollout_threads 100 --episode_length 50

python eval.py --env_name='SMLP' --experiment_name='Clearing_Function_Exp' --instance_id='01' --seed=1 --model_dir './results/SMLP/Clearing_Function_Exp/ins-01_lam-0.2_seed-1/models/' --n_eval_rollout_threads 100 --episode_length 50

python eval.py --env_name='SMLP' --experiment_name='Clearing_Function_Poly' --instance_id='01' --seed=1 --model_dir './results/SMLP/Clearing_Function_Poly/ins-01_lam-0.2_seed-1/models/' --n_eval_rollout_threads 100 --episode_length 50

python eval.py --env_name='SMLP_SCHED' --experiment_name='Flow_Shop' --instance_id='01' --seed=2 --model_dir './results/SMLP_SCHED/Flow_Shop/ins-01_lam-0.0_seed-1/models/' --n_eval_rollout_threads 100 --episode_length 100
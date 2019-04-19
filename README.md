# free_adv_train
To train a new robust model for free! run the following command specifying the replay parameter $m$:

python free_train.py -m 8

To evaluate a robust model using PGD with 2 random restarts run:

python multi_restart_pgd_attack.py --model_dir $MODEL_DIR --num_restarts 2

#!/bin/bash

python model.py  -experiment_name demo -exchange_samples 0 -model_type Fixed -max_exchange 3 -batch_size 32 -rec_w_dim 32 -sender_out_dim 32 -img_h_dim 256 -rec_hidden 64 -learning_rate 1e-4 -entropy_rec 0.01 -entropy_sen 0.01 -entropy_s 0.08 -use_binary -max_epoch 100 -top_k_dev 6 -top_k_train 6 -descr_train ./utils/descriptions.csv -descr_dev ./utils/descriptions.csv -train_file ./utils/train.hdf5 -dev_file ./utils/dev.hdf5  -wv_dim 100 -glove_path /scratch/lhg256/comms/glove/glove.6B.100d.txt -log_path /scratch/lhg256/comms/logs

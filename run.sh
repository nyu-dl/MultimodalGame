#!/bin/bash

python model_symmetric.py  -experiment_name test_training -exchange_samples 0 -model_type Fixed -max_exchange 1 -batch_size 4 -batch_size_dev 8 -m_dim 32 -h_dim 100 -desc_dim 100 -num_classes 10 -learning_rate 1e-4 -entropy_agent1 0.01 -entropy_agent2 0.01 -use_binary -max_epoch 100 -top_k_dev 3 -top_k_train 3 -dataset_path ./ShapeWorld/data/oneshape_simple_textselect -dataset_name oneshape_simple_textselect -dataset_size 250  -wv_dim 100 -glove_path ./glove.6B/glove.6B.100d.txt -log_path ./logs -debug_log_level DEBUG

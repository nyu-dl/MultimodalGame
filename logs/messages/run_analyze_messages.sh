#!/bin/bash

declare -a files=(
chains_msg/chain_331033_eval_only_A_1_17_message_stats.pkl
chains_msg/chain_331033_eval_only_A_1_20_message_stats.pkl
chains_msg/chain_331033_eval_only_A_1_2_message_stats.pkl
chains_msg/chain_331033_eval_only_A_1_4_message_stats.pkl
chains_msg/chain_331033_eval_only_A_1_7_message_stats.pkl
chains_msg/chain_331033_eval_only_A_4_5_message_stats.pkl
chains_msg/dense_331033_eval_only_A_1_17_message_stats.pkl
chains_msg/dense_331033_eval_only_A_1_20_message_stats.pkl
chains_msg/dense_331033_eval_only_A_1_2_message_stats.pkl
chains_msg/dense_331033_eval_only_A_1_4_message_stats.pkl
chains_msg/dense_331033_eval_only_A_1_7_message_stats.pkl
chains_msg/dense_331033_eval_only_A_4_5_message_stats.pkl
)

for i in "${files[@]}"
do
  echo $i
  python ../../analyze_messages.py --path $i > $i.txt
done

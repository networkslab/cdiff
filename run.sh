#!/bin/bash


python3 main.py --batch_size 32 --update_freq 1 --lr 0.001 --epochs 1000 --eval_every 2 --check_every 2 --diffusion_steps 200 --gamma 0.99 --log_home . --dataset taobao --dataset_dir ./data/taobao --transformer_dim 32 --transformer_heads 2 --num_encoder_layers 1 --dim_feedforward 64 --num_decoder_layers 1 --scheduler cosanneal --num_samples 5 --boxcox


for i in {1..5}; do
    python3 main.py --batch_size 32 --update_freq 1 --lr 0.0025 --epochs 1000 --eval_every 2 --check_every 2 --diffusion_steps 200 --gamma 0.99 --log_home . --dataset amazon --dataset_dir ./data/amazon --transformer_dim 32 --transformer_heads 2 --num_encoder_layers 1 --dim_feedforward 64 --num_decoder_layers 1 --scheduler cosanneal --num_samples 5 --boxcox --tgt_len 20 --seed $i
done
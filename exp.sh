for i in {1..5}; do
    python3 main.py --batch_size 2048 --update_freq 1 --lr 0.0005 --epochs 500 --eval_every 50 --check_every 2 --diffusion_steps 200 --gamma 0.99 --log_home . --dataset taobao --dataset_dir /home/jonus/tpp/data/taobao --transformer_dim 32 --transformer_heads 2 --num_encoder_layers 1 --dim_feedforward 64 --num_decoder_layers 1 --scheduler cosanneal --num_samples 7 --boxcox --tgt_len 20 --seed $i
done

for i in {1..5}; do
    python3 main.py --batch_size 128 --update_freq 1 --lr 0.0005 --epochs 500 --eval_every 50 --check_every 2 --diffusion_steps 100 --gamma 0.99 --log_home . --dataset taxi --dataset_dir /home/jonus/tpp/data/taxi --transformer_dim 32 --transformer_heads 2 --num_encoder_layers 1 --dim_feedforward 32 --num_decoder_layers 1 --scheduler cosanneal --num_samples 7 --boxcox --tgt_len 20 --seed $i
done




for i in {1..5}; do
    python3 main.py --batch_size 2048 --update_freq 1 --lr 0.0005 --epochs 500 --eval_every 50 --check_every 2 --diffusion_steps 200 --gamma 0.99 --log_home . --dataset stackoverflow --dataset_dir /home/jonus/tpp/data/so --transformer_dim 32 --transformer_heads 2 --num_encoder_layers 1 --dim_feedforward 64 --num_decoder_layers 1 --scheduler cosanneal --num_samples 7 --boxcox --tgt_len 20 --seed $i
done




for i in {1..5}; do
    python3 main.py --batch_size 1024 --update_freq 1 --lr 0.0005 --epochs 500 --eval_every 50 --check_every 2 --diffusion_steps 200 --gamma 0.99 --log_home . --dataset retweet --dataset_dir /home/jonus/tpp/data/retweet --transformer_dim 32 --transformer_heads 2 --num_encoder_layers 1 --dim_feedforward 64 --num_decoder_layers 1 --scheduler cosanneal --num_samples 7 --boxcox --tgt_len 20 --seed $i
done




for i in {1..5}; do
    python3 main.py --batch_size 1024 --update_freq 1 --lr 0.0005 --epochs 500 --eval_every 50 --check_every 2 --diffusion_steps 200 --gamma 0.99 --log_home . --dataset amazon --dataset_dir /home/jonus/tpp/data/amazon --transformer_dim 32 --transformer_heads 2 --num_encoder_layers 1 --dim_feedforward 64 --num_decoder_layers 1 --scheduler cosanneal --num_samples 7 --boxcox --tgt_len 20 --seed $i
done
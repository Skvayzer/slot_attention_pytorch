git pull
cd ./src/sa_autoencoder/
python3 train.py --mode "tetrominoes" --path_to_dataset "/content/slot_attention_pytorch/datasets/tetrominoes" --path_to_checkpoint "/content/slot_attention_pytorch/src/sa_autoencoder/ckpt/epoch=509-step=477870.ckpt" --device 0 --batch_size 64 --max_epochs 534 --seed 1


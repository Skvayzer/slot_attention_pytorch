git pull
cd ./src/sa_autoencoder/
python3 train.py --mode "multi_dsprites" --path_to_dataset "/home/alexandr_ko/datasets/multi_objects/multi_dsprites" --device 0 --batch_size 64 --max_epochs 534 --seed 7


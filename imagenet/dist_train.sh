
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
N_GPUS=6
PORT=23365

export num_of_gpus=$N_GPUS


python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train.py # --ckpt output/checkpoint_39.pth


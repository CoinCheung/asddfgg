
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1
N_GPUS=2
PORT=23365

export num_of_gpus=$N_GPUS


python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train.py


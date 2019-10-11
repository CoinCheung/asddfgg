
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
N_GPUS=4
PORT=23365
ckpt=output/checkpoint_135.pth

export num_of_gpus=$N_GPUS

python -c 'import torch; print(torch.__version__)'

python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train.py #--ckpt $ckpt
# python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT play.py #--ckpt $ckpt


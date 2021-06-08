
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
N_GPUS=8
PORT=23365

python -c 'import torch; print(torch.__version__)'

# cfg_file='./config/seti/resnet50.py'
cfg_file='./config/resnet50.py'


python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train.py --config $cfg_file

# python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train_cdata.py


# python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT eval.py

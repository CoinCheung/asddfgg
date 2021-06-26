
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
N_GPUS=8
PORT=43365

python -c 'import torch; print(torch.__version__)'

cfg_file='./config/seti/resnet50.py'
cfg_file='./config/seti/resnet50_adamw_warmup10.py'
cfg_file='./config/seti/timm_r18d.py'

ckpt_path='./res/model_final_naive_5.pth'


python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT eval.py --config $cfg_file --ckpt $ckpt_path

# python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train_cdata.py


# python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT eval.py

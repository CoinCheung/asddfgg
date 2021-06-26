
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
N_GPUS=8
# export CUDA_VISIBLE_DEVICES=0
# N_GPUS=1
PORT=23367

python -c 'import torch; print(torch.__version__)'

# cfg_file='./config/seti/resnet50_mixup.py'
cfg_file='./config/seti/timm_r18d.py'
# cfg_file='./config/seti/resnet50.py'


# python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train.py --config $cfg_file
# python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train.py --config $cfg_file
# python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train.py --config $cfg_file
# python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train.py --config $cfg_file
# python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train.py --config $cfg_file
#


# cfg_file='./config/seti/resnet50_adamw_warmup5.py'
#
# python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train.py --config $cfg_file
# python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train.py --config $cfg_file
# python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train.py --config $cfg_file
# python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train.py --config $cfg_file
# python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train.py --config $cfg_file
#
#
# cfg_file='./config/seti/resnet50_adamw_warmup10.py'
#
# python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train.py --config $cfg_file
# python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train.py --config $cfg_file
# python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train.py --config $cfg_file
# python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train.py --config $cfg_file
# python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train.py --config $cfg_file

# cfg_file='./config/seti/resnet50_adamw_warmup10.py'
cfg_file='./config/seti/timm_r18d.py'
# cfg_file='./config/seti/timm_effnet_b1.py'
for f_ind in $(seq 1 1 5);
do
    CURR=`pwd`
    cd datasets/seti/
    rm train.txt val.txt
    ln -s folds/train_fold_$f_ind.txt ./train.txt
    ln -s folds/val_fold_$f_ind.txt ./val.txt
    cd $CURR

    python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train.py --config $cfg_file
    cd res
    mv model_final_naive.pth model_final_naive_$f_ind.pth
    mv model_final_ema.pth model_final_ema_$f_ind.pth
    cd $CURR
done


# python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train_cdata.py


# python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT eval.py



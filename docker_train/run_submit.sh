

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NGPUS=8
PORT=42345


for f in `ls res | grep done`;
do
    rm res/$f
done
python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port=$PORT ./gen_submit.py



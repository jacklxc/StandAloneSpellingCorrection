source activate paddle
module load cuda cudnn nccl

set -eux

export MODEL_PATH=../ERNIE/ERNIE_weight
export TASK_DATA_PATH=../misspelling/data
export FLAGS_eager_delete_tensor_gb=0
export FLAGS_sync_nccl_allreduce=1
export PYTHONPATH=./ernie:${PYTHONPATH:-}
python -u $1

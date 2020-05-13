import sys
import re
import os
import json
import glob
import argparse
import subprocess
def build_command(dependency):
    """
    """
    n_gpu = 10
    job_name = "ERNIE_vanilla"
    partition = "2080Ti_mlong"
    cmd_str = 'python -u ./ernie/run_sequence_labeling.py \
	--use_cuda true \
	--do_train true \
	--do_val true \
	--do_test false \
	--batch_size 8 \
	--init_checkpoint ./checkpoints/vanilla_natural/latest \
	--num_labels 100005 \
	--chunk_scheme "IOB" \
	--label_map_config ${TASK_DATA_PATH}/full_label_map.json \
	--train_set ${TASK_DATA_PATH}/train_ernie.tsv \
	--dev_set ${TASK_DATA_PATH}/dev_ernie.tsv,${TASK_DATA_PATH}/test_ernie.tsv \
	--test_set ${TASK_DATA_PATH}/test_ernie.tsv \
	--vocab_path ${MODEL_PATH}/vocab.txt \
	--ernie_config_path ${MODEL_PATH}/ernie_config.json \
	--checkpoints ./checkpoints/vanilla_natural \
	--save_steps 2000 \
	--weight_decay 0.01 \
	--warmup_proportion 0.0 \
	--validation_steps 1000 \
	--epoch 200 \
	--learning_rate 5e-5 \
    --skip_steps 100 \
	--use_fp16 false \
	--max_seq_len 256 \
	--num_iteration_per_drop_scope 1 \
	--for_cn false \
	--test_save ./ernie_corrector_prediction/vanilla_ernie'

    dependency_str = ''
    if dependency is not None:
        dependency_str = '--dependency afterany:{} '.format(dependency)

    command = ('sbatch -N 1 '
               '--job-name={job_name} '
               '--ntasks 1 '
               '--gres=gpu:{n_gpu} '
               '--signal=USR1@120 '
               '--wrap "srun stdbuf -i0 -o0 -e0 {cmd_str}" '
               '--partition={partition} '
               '{dependency_str} '.format(
                   n_gpu=n_gpu,
                   job_name=job_name,
                   cmd_str=cmd_str,
                   partition=partition,
                   dependency_str=dependency_str))
    return command
def run_slurm_command(command):
    """
    """
    output = subprocess.check_output(
        command, stderr=subprocess.STDOUT, shell=True).decode()
    job_id = re.search(r"[Jj]ob [0-9]+", output).group(0)
    job_id = int(job_id.split(' ')[1])
    print("JOB {} Submitted!".format(job_id))
    return job_id

def schedule():
    times = 3
    dependency = None
    for _ in range(times):
        command = build_command(dependency)
        dependency = run_slurm_command(command)

schedule()

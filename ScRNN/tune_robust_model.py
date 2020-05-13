import sys
import re
import os
import json
import glob
import argparse
import subprocess
def build_command(dependency, lr, hidden_dim):
    """
    """
    n_gpu = 10
    job_name = "robust_misspelling"
    partition = "2080Ti"
    gpu_str = ",".join([str(i) for i in range(n_gpu)])
    cmd_str = \
        'python -m paddle.distributed.launch --selected_gpus={} --log_dir ./robust_log/lr_{}_hidden_{} robust_model.py --lr {} --hidden_size {}'.format(
            gpu_str, lr, hidden_dim, lr, hidden_dim)

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
    lrs = [0.001]
    hidden_dims= [650]
    times = 4
    for lr in lrs:
        for hidden_dim in hidden_dims:
            dependency = None
            for _ in range(times):
                command = build_command(dependency, lr, hidden_dim)
                dependency = run_slurm_command(command)
parser = argparse.ArgumentParser(description='Arguments Parser')
args = parser.parse_args()
schedule()
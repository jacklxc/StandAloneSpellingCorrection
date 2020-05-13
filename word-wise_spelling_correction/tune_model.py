import sys
import re
import os
import json
import glob
import argparse
import subprocess
def build_command(dependency, config_file):
    """
    """
    n_gpu = 10
    job_name = config_file.split(".")[0]
    partition = "2080Ti"
    gpu_str = ",".join([str(i) for i in range(n_gpu)])
    cmd_str = \
        'python -m paddle.distributed.launch --selected_gpus={} --log_dir ./latest_log/{} model.py {}'.format(
            gpu_str, job_name, config_file)

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
    config_prefix = "configs/"
    config_suffix = "_config.json"
    configs = ["standard", "standard_char", "standard_mix"]
    times = 4

    for config in configs:
        config_file = config_prefix + config + config_suffix
        dependency = None
        for _ in range(times):
            command = build_command(dependency, config_file)
            dependency = run_slurm_command(command)
parser = argparse.ArgumentParser(description='Arguments Parser')
args = parser.parse_args()
schedule()
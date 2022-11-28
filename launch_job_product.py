import argparse
import os
import subprocess
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR='./logs'
MEM='48G'
GPU_TYPE='rtx_2080'
GPUS=1

def runcmd(model, model_name, task, seed, duration, no_config):
    if duration == '1:00:00':
        partition = 'allcs'
    else:
        partition = 'pnlp'
    cmd = rf'''sbatch \
            -A {partition} \
            --exclude=node030,node031,node003 \
            -J {model_name}-{task}-{seed} \
            --time {duration} \
            --gres gpu:{GPU_TYPE}:{GPUS} \
            --mem {MEM} \
            -N 1 \
            -n 1 \
            --output {OUTPUT_DIR}/glue-finetuning-{model_name}-{task}-{seed}-%J.log \
            python glue_eval/launch_glue.py {model} {task} {seed}'''
    if no_config:
        cmd += ' --no_config'
    cmd += ' &'
    subprocess.run(cmd, shell=True)


def main():
    models = {
            'roberta-base',
            'roberta-large',
            }
    seeds = [
            0,
            1,
            2,
            ]
    task_duration = {
            'CoLA': '1:00:00',
            'SST2': '4:00:00',
            'MRPC': '1:00:00',
            'STSB': '1:00:00',
            'QQP': '2-00:00:00',
            'MNLI': '2-00:00:00',
            'QNLI': '5:00:00',
            'RTE': '1:00:00',
            'WNLI': '1:00:00',
            }
    for task, duration in task_duration.items():
        for seed in seeds:
            for model in models:
                if not Path(model).is_file() or Path(model).is_dir():
                    model_name = model.replace('/', '_')
                    runcmd(model, model_name, task, seed, duration, no_config=True)
                else:
                    model_name = Path(model).stem
                    runcmd(model, model_name, task, seed, duration, no_config=False)


if __name__ == '__main__':
    main()


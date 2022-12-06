import subprocess
import os
from pathlib import Path
import argparse


def main(model_name_or_path, task, seed, no_config):
    key = Path(model_name_or_path).stem
    pyfile = Path(__file__).absolute().parent.joinpath('run_glue.py').as_posix()
    output_dir = Path(__file__).absolute().parent.parent.joinpath(f'output/{key}/{task}/seed_{seed}/').as_posix()
    cmd = rf"""
    export WANDB_ENTITY=princeton-nlp
    export WANDB_PROJECT=angry-bert-{task}
    export TASK_NAME={task}
    
    python {pyfile} \
            --model_name_or_path {model_name_or_path} \
            --task_name {task} \
            --run_name {key}/seed-{seed} \
            --do_train \
            --do_eval \
            --seed {seed} \
            --save_strategy epoch \
            --learning_rate 2e-5 \
            --num_train_epochs 6 \
            --output_dir {output_dir} \
            --overwrite_output_dir \
            --logging_steps 50 \
            --evaluation_strategy epoch \
            --load_best_model_at_end \
            --save_steps -1 \
            --save_total_limit 1 \
            --fp16 """
    if not no_config:
        cmd += r""" \
            --max_seq_length 128 \
            --tokenizer_name roberta-base \
            --config_name roberta-base """
    else:
        cmd += r""" \
            --max_seq_length 512 \
            --pad_to_max_length True """
    if 'large' in model_name_or_path or 'multibert' in model_name_or_path:
        cmd += r""" \
            --per_device_train_batch_size 16 \
            --gradient_accumulation_steps 2 """
    else:
        cmd += r""" \
            --per_device_train_batch_size 32 """
    if 'multibert' in model_name_or_path:
        cmd += r""" \
                --overwrite_cache True """

    print(cmd)
    subprocess.run(cmd, shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name_or_path', type=str, help='Model name or path')
    parser.add_argument('task', type=str,  help='GLUE task to evaluate', choices=['CoLA', 'SST2', 'MRPC', 'STSB', 'QQP', 'MNLI', 'QNLI', 'RTE', 'WNLI'])
    parser.add_argument('seed', type=int,  help='Seed to use')
    parser.add_argument('--no_config', action='store_true', help='Use for huggingface keys (e.g. roberta-large)')
    args = parser.parse_args()
    main(**vars(args))

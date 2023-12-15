#!/bin/bash

python main.py --task_name cheetah_run
python main.py --task_name walker_stand
python main.py --task_name walker_stand


wait

python main.py --task_name walker_stand
python main.py --task_name walker_stand
python main.py --task_name walker_stand
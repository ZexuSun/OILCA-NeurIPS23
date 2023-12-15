import torch
import argparse
import torch.nn as nn
import numpy as np
from model_ori import LobsDICE
import os
import random
from tqdm import tqdm


def arg_parser():
    parser = argparse.ArgumentParser(description='LibsDICE')
    parser.add_argument('--data_path', type=str, default='/home/sunzexu/Causal_RL/OILCE/dataset/dm_control_suite/', metavar='G',
                        help='data path of offline dataset')
    parser.add_argument('--expert-traj-path', metavar='G', default='/home/sunzexu/Causal_RL/OILCE/dataset/dm_control_suite/cheetah_run/expert_cheetah_run_dataset_episode.npy')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epoch-num', type=int, default=1)
    parser.add_argument('--max-iter-num', type=int, default=1)
    parser.add_argument('--save-model-interval', type=int, default=1)

    parser.add_argument('--critic_lr', default=3e-4, type=float)
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--grad_reg_coeffs', default=(0.1, 1e-4))
    parser.add_argument('--observation_dim', default=18, type=int)
    parser.add_argument('--action_dim', default=6, type=int)
              
    parser.add_argument('--task_name', type=str, default='cheetah_run', metavar='N',
                        help='task name of deepmind control suite')  
    args = parser.parse_args()

    return args


if __name__=='__main__':
    args = arg_parser()
    imitator = LobsDICE(args.observation_dim, args.action_dim, args=args)
    expert_traj = np.load(args.data_path + args.task_name + '/expert_{}_dataset_episode.npy'.format(args.task_name)).astype(np.float64)
    union_traj = np.load(args.data_path + args.task_name + '/total_{}_dataset_episode.npy'.format(args.task_name)).astype(np.float64)


    expert_abso_state_mask = np.zeros([expert_traj.shape[0], expert_traj.shape[1]-1, 1], dtype=np.float32)
    expert_abso_next_state_mask = np.zeros([expert_traj.shape[0], expert_traj.shape[1]-1, 1], dtype=np.float32)
    expert_abso_next_state_mask[:, -1, 0] = 1.0

    union_abso_state_mask = np.zeros([union_traj.shape[0], union_traj.shape[1]-1, 1], dtype=np.float32)
    union_abso_next_state_mask = np.zeros([union_traj.shape[0], union_traj.shape[1]-1, 1], dtype=np.float32)
    union_abso_next_state_mask[:, -1, 0] = 1.0


    expert_traj_train_dict = {
            'observations': torch.from_numpy(np.c_[expert_traj[:, :-1, :17], expert_abso_state_mask]).reshape(-1, args.observation_dim),
            'actions': torch.from_numpy(expert_traj[:, 1:, 17:23]).reshape(-1, args.action_dim),
            'next_observations': torch.from_numpy(np.c_[expert_traj[:, 1:, :17], expert_abso_next_state_mask]).reshape(-1, args.observation_dim),
        }
    init_states = union_traj[:, 0, :17]
    union_traj_train_dict = {
            'init_observations': torch.from_numpy(np.c_[init_states, np.zeros([init_states.shape[0], 1], dtype=np.float32)]).reshape(-1, args.observation_dim),
            'observations': torch.from_numpy(np.c_[union_traj[:, :-1, :17], union_abso_state_mask]).reshape(-1, args.observation_dim),
            'actions': torch.from_numpy(union_traj[:, 1:, 17:23]).reshape(-1, args.action_dim),
            'next_observations': torch.from_numpy(np.c_[union_traj[:, 1:, :17], union_abso_next_state_mask]).reshape(-1, args.observation_dim)
        }

    expert_states = expert_traj_train_dict['observations'].float().cuda()
    expert_next_states = expert_traj_train_dict['next_observations'].float().cuda()
    union_init_states = union_traj_train_dict['init_observations'].float().cuda()
    union_states = union_traj_train_dict['observations'].float().cuda()
    union_actions = union_traj_train_dict['actions'].float().cuda()
    union_next_states = union_traj_train_dict['next_observations'].float().cuda()

    

    for epoch in tqdm(range(args.epoch_num)):
        for step in range(args.max_iter_num):
            union_init_indices = np.random.randint(0, len(union_init_states), size=args.batch_size)
            expert_indices = np.random.randint(0, len(expert_states), size=args.batch_size)
            union_indices = np.random.randint(0, len(union_states), size=args.batch_size)
            info_dict = imitator.train_model(
                union_init_states[union_init_indices],
                expert_states[expert_indices],
                expert_next_states[expert_indices],
                union_states[union_indices],
                union_actions[union_indices],
                union_next_states[union_indices]
            )

        for key, val in info_dict.items():
            print(f'{key:25}: {val:8.3f}')
        print('===========')
        if (epoch+1) % args.save_model_interval == 0:
            torch.save(imitator.actor.state_dict(), 'learned_models/LobsDICE_{}_{}.pkl'.format(args.task_name, 
                        int(epoch / args.save_model_interval)))
    


    
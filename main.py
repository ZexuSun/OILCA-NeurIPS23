import numpy as np
import torch
import argparse
import os
import time
from tqdm import tqdm

from algos.augment_DWBC_expert import DWBC
from dm_control import suite
import warnings
warnings.filterwarnings("ignore")


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../dataset/dm_control_suite/')
    parser.add_argument('--env', default="cheetah_run")  # environment name
    parser.add_argument("--seed", default=42, type=int)  # Sets PyTorch and Numpy seeds # 20.  42 is better
    parser.add_argument("--max_timesteps", default=1000, type=int)  # Max time steps to run environment
    parser.add_argument('--epoch-num', type=int, default=200)
    # DWBC
    parser.add_argument('--observation_dim', default=17, type=int)
    parser.add_argument('--action_dim', default=6, type=int)
    parser.add_argument("--batch_size", default=512, type=int)  # Batch size for both actor and critic
    parser.add_argument("--alpha", default=7.5, type=float)
    parser.add_argument("--no_pu", default=False, type=bool)
    parser.add_argument("--eta_change", default=True, type=bool)
    parser.add_argument("--aug_percent", default=0.5, type=float)
    parser.add_argument("--eta_init", default=0.3, type=float)
    parser.add_argument("--eta_dw", default=0.2, type=float)
    parser.add_argument("--eta_up", default=0.8, type=float)
    parser.add_argument('--d_update_num', type=int, default=100)
    parser.add_argument("--aug_steps", default=1, type=int)
    parser.add_argument("--no_normalize", default=False, type=bool)
    parser.add_argument('--expert-policy-path', metavar='G', default='../learned_models/BC_all/bc_model_cheetah_run_9.pkl',
                    help='path of the expert trajectories')
    

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = arg_parser()
    eps = 1e-3
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    state_dim = args.observation_dim
    action_dim = args.action_dim

    # Initialize policy
    policy = DWBC(state_dim, action_dim, args)

    # Load dataset
    
    expert_traj = np.load(args.data_path + args.env + '/expert_{}_dataset_episode.npy'.format(args.env)).astype(np.float32)
    unlabel_traj = np.load(args.data_path + args.env + '/unlabeled_{}_dataset_episode.npy'.format(args.env)).astype(np.float32)



    expert_traj_train_dict = {
            'observations': torch.from_numpy(expert_traj[:, :-1, :17]).reshape(-1, args.observation_dim),
            'actions': torch.from_numpy(expert_traj[:, :-1, 17:23]).reshape(-1, args.action_dim),
            'next_observations': torch.from_numpy(expert_traj[:, 1:, :17]).reshape(-1, args.observation_dim)
        }
    unlabel_traj_train_dict = {
            'observations': torch.from_numpy(unlabel_traj[:, :-1, :17]).reshape(-1, args.observation_dim),
            'actions': torch.from_numpy(unlabel_traj[:, :-1, 17:23]).reshape(-1, args.action_dim),
            'next_observations': torch.from_numpy(unlabel_traj[:, 1:, :17]).reshape(-1, args.observation_dim)
        }

    states_e = expert_traj_train_dict['observations'].float().cuda()
    action_e = expert_traj_train_dict['actions'].float().cuda()
    next_states_e = expert_traj_train_dict['next_observations'].float().cuda()

    states_o = unlabel_traj_train_dict['observations'].float().cuda()
    action_o = unlabel_traj_train_dict['actions'].float().cuda()
    next_states_o = unlabel_traj_train_dict['next_observations'].float().cuda()
    
    states_b = torch.cat([states_e, states_o], dim=0)

    print('# {} of expert demonstraions'.format(states_e.shape[0]))
    print('# {} of imperfect demonstraions'.format(states_o.shape[0]))


    if args.no_normalize:
        shift, scale = 0, 1
    else:
        shift = torch.mean(states_b, 0)
        scale = torch.std(states_b, 0) + 1e-3

    
    states_e = (states_e - shift) / scale
    states_o = (states_o - shift) / scale

    next_states_e = (next_states_e - shift) / scale
    next_states_o = (next_states_o - shift) / scale


    # Start training
    epoch_returns = []
    epoch_success = []
    epoch_steps = []
    for epoch in tqdm(range(args.epoch_num)):
        for t in range(int(args.max_timesteps)):

            batch_e_idx = np.random.randint(0, len(states_e), size=args.batch_size)
            batch_o_idx = np.random.randint(0, len(states_o), size=args.batch_size)
            batch_state_e = states_e[batch_e_idx]
            batch_action_e = action_e[batch_e_idx]
            batch_next_states_e = next_states_e[batch_e_idx]
            batch_state_o = states_o[batch_o_idx]
            batch_action_o = action_o[batch_o_idx]
            batch_next_states_o = next_states_o[batch_o_idx]
            
            policy.train(batch_state_e, batch_action_e, batch_next_states_e, batch_state_o, batch_action_o, shift, scale)

        # online test
        env = suite.load(args.env.split('_')[0], args.env.split('_')[1])
        all_returns = []
        all_success = []
        all_steps = []
        for _ in range(10):
            step = 0
            time_step = env.reset()
            #print('time_step', time_step)
            #print('time_step.last()', time_step.last())
            returns = 0
            success_flag = 0
            while not time_step.last() and step < 1000:
                if not args.no_normalize:
                    action, _ , _= policy.policy.forward((torch.concat([torch.tensor(time_step.observation['position']).cuda().float(), torch.tensor(time_step.observation['velocity']).cuda().float()])-shift)/scale) # cheetah_run
                    #action, _, _ = policy.policy.forward((torch.concat([torch.tensor([time_step.observation['height']]).cuda().float(), torch.tensor(time_step.observation['orientations']).cuda().float(), torch.tensor(time_step.observation['velocity']).cuda().float()])-shift)/) # walker_stand
                    #action, _, _ = policy.policy.forward((torch.concat([torch.tensor(time_step.observation['joint_angles']).cuda().float(), torch.tensor(time_step.observation['target']).cuda().float(), torch.tensor([time_step.observation['upright']]).cuda().float(), torch.tensor(time_step.observation['velocity']).cuda().float()])-shift)/) # fish_swim
                else:
                    action, _ , _= policy.policy.forward(torch.concat([torch.tensor(time_step.observation['position']).cuda().float(), torch.tensor(time_step.observation['velocity']).cuda().float()])) # cheetah_run
                    #action, _, _ = policy.policy.forward(torch.concat([torch.tensor([time_step.observation['height']]).cuda().float(), torch.tensor(time_step.observation['orientations']).cuda().float(), torch.tensor(time_step.observation['velocity']).cuda().float()])) # walker_stand
                    #action, _, _ = policy.policy.forward(torch.concat([torch.tensor(time_step.observation['joint_angles']).cuda().float(), torch.tensor(time_step.observation['target']).cuda().float(), torch.tensor([time_step.observation['upright']]).cuda().float(), torch.tensor(time_step.observation['velocity']).cuda().float()]).cuda().float()])) # fish_swim
                time_step = env.step(action.detach().cpu().numpy())
                if time_step.reward is not None:
                    reward = time_step.reward
                else:
                    reward = 0
                #print('reward', reward)
                returns +=  reward
                step += 1

            if step < 1000:
                success_flag = 1

            print(returns, end='\r')
            all_returns.append(returns)
            all_steps.append(step)
            all_success.append(success_flag)

        print(args.env, ' average return :', np.mean(all_returns))    
        print(args.env, ' average step :', np.mean(all_steps))   
        print(args.env, ' average success rate :', np.mean(all_success))  

        # records for each training epoch
        epoch_returns.append(np.mean(all_returns))
        epoch_steps.append(np.mean(all_steps))
        epoch_success.append(np.mean(all_success))

    # results to csv
    print('np.array(epoch_returns).shape', np.array(epoch_returns).shape)
    results_arrays = np.concatenate((np.array(epoch_returns).reshape(-1,1), np.array(epoch_steps).reshape(-1,1), np.array(epoch_success).reshape(-1,1)), axis=1)
    np.savetxt('../saved_results/' + args.env + '_OILCA' + '_seed_{}'.format(args.env) + '.csv', results_arrays, delimiter=',', header='epoch_return, epoch_step, epoch_success')

#test的时候也-mean/std

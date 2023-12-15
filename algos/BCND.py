import tqdm
import torch
from torch import nn
import pickle
import numpy as np
from torch import optim
from random import shuffle
from dotmap import DotMap
from torch.utils.data import Dataset, DataLoader
from model import TanhGaussianPolicy



class custom_data(Dataset):
    def __init__(self, data_cfg, model_config):
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg

        data_np = np.load(data_cfg.dataset_filepath)
        self.data = data_np.reshape(-1, data_np.shape[2])

        
    def __getitem__(self, index):
        state = torch.from_numpy(self.data[index, 0:model_cfg.observation_dim])
        action = torch.from_numpy(self.data[index, model_cfg.observation_dim:model_cfg.observation_dim+model_cfg.action_dim])
        return state, action

    def __len__(self):
        return len(self.data)


class BCND:
    def __init__(self, model_cfg, reward_model):
        self.model_cfg = model_cfg
        self.train_data_cfg = model_cfg.train_data_cfg
        self.val_data_cfg = model_cfg.val_data_cfg
        self.train_dataloader = DataLoader(custom_data(self.train_data_cfg, self.model_cfg), batch_size=model_cfg.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(custom_data(self.val_data_cfg, self.model_cfg), batch_size=model_cfg.batch_size, pin_memory=True)
        self.model = TanhGaussianPolicy(
            observation_dim=model_cfg.observation_dim,
            action_dim=model_cfg.action_dim,
            arch='128-128',
            no_tanh=True
        )
        self.reward_model = reward_model

    def train_and_val(self, epoch):
        loss_function = nn.MSELoss()
        self.model.train()
        if torch.cuda.is_available():
            self.model.cuda()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

        with tqdm.tqdm(total=len(self.train_dataloader)) as progress_bar:
            batch_cnt = 0
            loss_train = []
            for batch_idx, batch in enumerate(self.train_dataloader):
                progress_bar.update(1)
                batch_cnt += 1
                if torch.cuda.is_available():
                    for i in range(len(batch)):
                        batch[i] = batch[i].to('cuda')
                observation, action = batch

                observation = observation.reshape(-1, model_cfg.observation_dim).float()
                action = action.reshape(-1, model_cfg.action_dim).float()
                _, reward = self.reward_model(observation)

                action_log_prob = self.model.log_prob(observation, action)

                loss = -torch.mean(action_log_prob * reward) 
                optimizer.zero_grad() 
                loss.backward()
                loss_train.append(loss.item())
                # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            loss_val = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.val_dataloader):
                    batch_cnt += 1
                if torch.cuda.is_available():
                    for i in range(len(batch)):
                        batch[i] = batch[i].to('cuda')
                    observation, action = batch

                    observation = observation.reshape(-1, model_cfg.observation_dim).float()
                    action = action.reshape(-1, model_cfg.action_dim).float()
                    reward = self.reward_model(observation)

                    _, action_log_prob = self.model.log_prob(observation, action)
                    loss = -torch.mean(action_log_prob * reward) 
                    loss_val.append(loss.item())
                    # nn.utils.clip_grad_norm_(model.parameters(), 1.0)


            epoch_train_loss = sum(loss_train) / len(loss_train)
            epoch_val_loss = sum(loss_val) / len(loss_val)

            print("Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}".format(epoch+1, epoch_train_loss, epoch_val_loss))
        if (epoch+1) % self.model_cfg.save_model_interval == 0:
            torch.save(self.model.state_dict(), 'learned_models/bcnd_model_{}_{}.pkl'.format(model_cfg.task_name, int(epoch / self.model_cfg.save_model_interval)))
        """clean up gpu memory"""
        torch.cuda.empty_cache()



if __name__ == '__main__':
    train_data_cfg = DotMap({
        'dataset_filepath': 'dataset/dm_control_suite/cheetah_run/total_cheetah_run_dataset_episode_train.npy',
    })
    val_data_cfg = DotMap({
        'dataset_filepath': 'dataset/dm_control_suite/cheetah_run/total_cheetah_run_dataset_episode_test.npy',
    })
    model_cfg = DotMap({
        'task_name': 'cheetah_run',
        'train_data_cfg': train_data_cfg,
        'val_data_cfg': val_data_cfg,
        'epoch': 50,
        'batch_size': 128,
        'save_model_interval': 20,
        'observation_dim': 17,
        'action_dim': 6
    })
    reward_policy = TanhGaussianPolicy(
            observation_dim=model_cfg.observation_dim,
            action_dim=model_cfg.action_dim,
            arch='128-128',
            no_tanh=True
        ).cuda()
    reward_policy.load_state_dict(torch.load('/home/sunzexu/Causal_RL/OILCE/learned_models/bc_model_cheetah_run_0.pkl'))
    agent = BCND(model_cfg, reward_model=reward_policy)
    for epoch in range(model_cfg.epoch):
        agent.train_and_val(epoch)

    
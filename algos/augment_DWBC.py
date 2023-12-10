import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
from ivae_exogenous_model import iVAE
from main_ivae_model import vae_args
import joblib
from model import TanhGaussianPolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN_MIN = -9.0
MEAN_MAX = 9.0
LOG_STD_MIN = -5
LOG_STD_MAX = 2
LOG_PI_NORM_MAX = 10
LOG_PI_NORM_MIN = -20

EPS = 1e-7


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, action_dim)
        self.sigma_head = nn.Linear(256, action_dim)

    def _get_outputs(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        mu = self.mu_head(a)
        mu = torch.clip(mu, MEAN_MIN, MEAN_MAX)
        log_sigma = self.sigma_head(a)
        log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = TransformedDistribution(
            Normal(mu, sigma), TanhTransform(cache_size=1)
        )
        a_tanh_mode = torch.tanh(mu)
        return a_distribution, a_tanh_mode

    def forward(self, state):
        a_dist, a_tanh_mode = self._get_outputs(state)
        action = a_dist.rsample()
        logp_pi = a_dist.log_prob(action).sum(axis=-1)
        return action, logp_pi, a_tanh_mode

    def get_log_density(self, state, action):
        a_dist, action_pred = self._get_outputs(state)
        action_clip = torch.clip(action, -1. + EPS, 1. - EPS)
        logp_action = a_dist.log_prob(action_clip)
        return logp_action


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()

        self.fc1_1 = nn.Linear(state_dim + action_dim, 128)
        self.fc1_2 = nn.Linear(action_dim, 128)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action, log_pi):
        sa = torch.cat([state, action], 1)
        d1 = F.relu(self.fc1_1(sa))
        d2 = F.relu(self.fc1_2(log_pi))
        d = torch.cat([d1, d2], 1)
        d = F.relu(self.fc2(d))
        d = torch.sigmoid(self.fc3(d))
        d = torch.clip(d, 0.1, 0.9)
        return d


class DWBC(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            args
    ):

        self.policy_expert = TanhGaussianPolicy(
            observation_dim=state_dim,
            action_dim=action_dim,
            arch='128-128',
            no_tanh=True
        ).cuda()
        self.policy_expert.load_state_dict(torch.load(args.expert_policy_path))

        self.policy = Actor(state_dim, action_dim).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4, weight_decay=0.005)

        self.discriminator = Discriminator(state_dim, action_dim).to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)

        self.alpha = args.alpha
        self.no_pu_learning = args.no_pu
        self.d_update_num = args.d_update_num
        
        self.state_dim = args.observation_dim
        self.action_dim = args.action_dim

        self.task_name = args.env
        self.aug_steps = args.aug_steps


        self.total_it = 0
        self.max_timesteps = args.max_timesteps
        self.batch_size = args.batch_size

        self.eta = args.eta_init
        self.eta_up = args.eta_up
        self.eta_dw = args.eta_dw
        self.eta_change = args.eta_change
        self.aug_percent = args.aug_percent

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        _, _, action = self.policy(state)
        return action.cpu().data.numpy().flatten()

    def train(self, state_e, action_e, next_state_e, state_o, action_o, shift, scale):
        self.total_it += 1

        state_o_in, action_o_in = state_o.clone(), action_o.clone()    

        #Optimize Discriminator

        
        #data augmentation
#----------------------------
        if self.total_it % self.d_update_num == 0 and self.aug_steps != 0:
            ivae_arg = vae_args()
            # load iVAE model
            vae_model = iVAE(latent_dim=self.state_dim, state_dim=self.state_dim, action_dim=self.action_dim, aux_dim=ivae_arg.aux_dim, aux_max=ivae_arg.cluster_num,  
                            hidden_dim=ivae_arg.hidden_dim, anneal=ivae_arg.anneal) 
            vae_model.load_state_dict(torch.load('learned_models/ivae_' + self.task_name +'.pt'))
            vae_model.cuda()

            #cluster function
            cluster_file_path = '{}_cluster_{}.pkl'.format(self.task_name, ivae_arg.cluster_num)
            cluster = joblib.load(cluster_file_path)

            #auxilary variable
            u = torch.tensor(cluster.predict(np.float32(next_state_e.cpu().numpy()))).long().cuda()
            embed_u = vae_model.cat_embedding(u).cuda()

            for aug_step in range(self.aug_steps):
                aug_idx = np.random.randint(0, len(state_e), size=int(self.aug_percent * self.batch_size))
                next_state_e_vae = next_state_e * scale + shift
                state_e_vae = state_e * scale + shift
                action_e_vae = action_e
                aug_state_e, _ = vae_model.reconstruct(next_state_e_vae, embed_u, state_e_vae, action_e_vae)
                aug_state_e = (aug_state_e - shift) / scale
                aug_action_e, _ = self.policy_expert(aug_state_e)
                aug_state_e = aug_state_e[aug_idx]
                aug_action_e = aug_action_e[aug_idx]
                state_o = torch.cat([state_o_in, aug_state_e], axis=0).detach()
                action_o = torch.cat([action_o_in, aug_action_e], axis=0).detach()

            #sample
            o_idx = np.random.randint(0, len(state_o), size=self.batch_size)
            state_o = state_o[o_idx]
            action_o = action_o[o_idx]

            if self.eta_change == True:
                log_pi_o_aug = self.policy.get_log_density(state_o, action_o)
                log_pi_o_aug_clip = torch.clip(log_pi_o_aug, LOG_PI_NORM_MIN, LOG_PI_NORM_MAX)
                log_pi_o_aug_norm = (log_pi_o_aug_clip - LOG_PI_NORM_MIN) / (LOG_PI_NORM_MAX - LOG_PI_NORM_MIN)
                d_o_aug = self.discriminator(state_o, action_o, log_pi_o_aug_norm.detach())
                
                
                d_o_aug_clip = torch.squeeze(d_o_aug).detach()
                d_o_aug_clip[d_o_aug_clip <= 0.5] = 0.0
                d_o_aug_clip[d_o_aug_clip > 0.5] = 1.0
                # print(torch.sum(d_o_aug_clip))
                self.eta = torch.sum(d_o_aug_clip) / self.batch_size
                
                self.eta = torch.clip(self.eta, self.eta_dw, self.eta_up)
            
        
        
        log_pi_e = self.policy.get_log_density(state_e, action_e)
        log_pi_o = self.policy.get_log_density(state_o, action_o)

        # Compute discriminator loss
        log_pi_e_clip = torch.clip(log_pi_e, LOG_PI_NORM_MIN, LOG_PI_NORM_MAX)
        log_pi_o_clip = torch.clip(log_pi_o, LOG_PI_NORM_MIN, LOG_PI_NORM_MAX)
        log_pi_e_norm = (log_pi_e_clip - LOG_PI_NORM_MIN) / (LOG_PI_NORM_MAX - LOG_PI_NORM_MIN)
        log_pi_o_norm = (log_pi_o_clip - LOG_PI_NORM_MIN) / (LOG_PI_NORM_MAX - LOG_PI_NORM_MIN)

        d_e = self.discriminator(state_e, action_e, log_pi_e_norm.detach())
        d_o = self.discriminator(state_o, action_o, log_pi_o_norm.detach())

        if self.total_it % self.d_update_num != 0:
            d_o = self.discriminator(state_o, action_o, log_pi_o_norm.detach())
        if self.no_pu_learning:
            d_loss_e = -torch.log(d_e)
            d_loss_o = -torch.log(1 - d_o)
            d_loss = torch.mean(d_loss_e + d_loss_o)
        else:
            d_loss_e = -torch.log(d_e)
            d_loss_o = -torch.log(1 - d_o) / self.eta + torch.log(1 - d_e)
            d_loss = torch.mean(d_loss_e + d_loss_o)
       
        if self.total_it % self.d_update_num == 0:
            self.discriminator_optimizer.zero_grad()
            d_loss.backward()
            self.discriminator_optimizer.step()

        

        # Compute policy loss
        d_e_clip = torch.squeeze(d_e).detach()
        d_o_clip = torch.squeeze(d_o).detach()
        d_o_clip[d_o_clip < 0.5] = 0.0

        bc_loss = -torch.sum(log_pi_e, 1)
        corr_loss_e = -torch.sum(log_pi_e, 1) * (self.eta / (d_e_clip * (1.0 - d_e_clip)) + 1.0)
        corr_loss_o = -torch.sum(log_pi_o, 1) * (1.0 / (1.0 - d_o_clip) - 1.0)
        p_loss = self.alpha * torch.mean(bc_loss) - torch.mean(corr_loss_e) + torch.mean(corr_loss_o)
        # Optimize the policy
        self.policy_optimizer.zero_grad()
        p_loss.backward()
        self.policy_optimizer.step()

    def save(self, filename):
        torch.save(self.discriminator.state_dict(), filename + "_discriminator")
        torch.save(self.discriminator_optimizer.state_dict(), filename + "_discriminator_optimizer")

        torch.save(self.policy.state_dict(), filename + "_policy")
        torch.save(self.policy_optimizer.state_dict(), filename + "_policy_optimizer")

    def load(self, filename):
        self.discriminator.load_state_dict(torch.load(filename + "_discriminator"))
        self.discriminator_optimizer.load_state_dict(torch.load(filename + "_discriminator_optimizer"))

        self.policy.load_state_dict(torch.load(filename + "_policy"))
        self.policy_optimizer.load_state_dict(torch.load(filename + "_policy_optimizer"))

import torch
import torch.nn as nn
import numpy as np 
from utils import TanhGaussianPolicy
from critic import Critic, Discriminator
from torch import optim
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import random
EPS = np.finfo(np.float32).eps
EPS2 = 1e-3

#without init state in the input
class LobsDICE(nn.Module):
    #init 
    def __init__(self, state_dim, action_dim, args):
        super(LobsDICE, self).__init__()

        critic_lr = args.critic_lr
        actor_lr = args.actor_lr


        self.cost = Critic(state_dim*2).cuda()
        self.nu = Critic(state_dim).cuda()
        self.mu = Critic(state_dim).cuda()
        self.discriminator = Discriminator(state_dim).cuda
        
        self.actor = TanhGaussianPolicy(state_dim, action_dim).cuda()


        self.cost_optimizer = optim.Adam(self.cost.parameters(), lr=critic_lr)
        self.critic_optimizer = optim.Adam([{'params': self.nu.parameters(), 'lr': critic_lr},
                                            {'params': self.mu.parameters(), 'lr': critic_lr}]
                                            )
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.grad_reg_coeffs = args.grad_reg_coeffs
        self.discount = args.gamma
        self.non_expert_regularization = args.alpha + 1 


    def train_model(self, expert_states, expert_next_states, imperfect_states,
                    imperfect_actions, imperfect_next_states):
        
        self.cost_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        self.cost_optimizer.zero_grad()

        expert_inputs = torch.cat([expert_states, expert_next_states], dim=-1)
        imperfect_inputs = torch.cat([imperfect_states, imperfect_next_states], dim=-1)

        expert_cost_val = self.cost(expert_inputs)
        imperfect_cost_val = self.cost(imperfect_inputs)
        unif_rand = torch.random(expert_states.shape[0], 1)
        mixed_inputs1 = unif_rand * expert_inputs + (1 - unif_rand) * imperfect_inputs
        mixed_inputs2 = unif_rand * random.shuffle(imperfect_inputs) + (1 - unif_rand) * imperfect_inputs
        mixed_inputs = torch.cat([mixed_inputs1, mixed_inputs2], dim=0)
        
        #gradient penalty
        with torch.no_grad():
            cost_output = self.cost(mixed_inputs)
            cost_output = torch.math.log(1 / (nn.sigmoid(cost_output) + EPS2) - 1 + EPS2)


        cost_mix_grad = torch_grad(outputs=cost_output, inputs=interpolated,
                               grad_outputs=torch.ones(cost_output.size()).cuda() if self.use_cuda else torch.ones(
                               cost_output.size()),
                               create_graph=True, retain_graph=True)[0] + EPS
        cost_mix_grad.view(cost_output.shape[0], -1)
        cost_grad_penalty = torch.sqrt(torch.sum(cost_mix_grad ** 2, dim=-1))
        cost_loss = torch.mean(self.discriminator(imperfect_cost_val))- torch.mean(self.discriminator(expert_cost_val)) \
                        + self.grad_reg_coeffs[0] * cost_grad_penalty

        expert_cost = torch.math.log(1 / (nn.sigmoid(expert_cost_val) + EPS2) - 1 + EPS2)
        imperfect_cost = torch.math.log(1 / (nn.sigmoid(imperfect_cost_val) + EPS2) - 1 + EPS2)


         # nu learning
        # init_nu = self.nu(init_states)
        expert_mu = self.mu(expert_inputs)
        expert_nu = self.nu(expert_states)
        expert_next_nu = self.nu(expert_next_states)
        imperfect_mu = self.mu(imperfect_inputs)
        imperfect_nu = self.nu(imperfect_states)
        imperfect_next_nu = self.nu(imperfect_next_states)

        imperfect_adv_mu_r = imperfect_mu - imperfect_cost
        imperfect_adv_mu_nu = self.discount * imperfect_next_nu - imperfect_mu - imperfect_nu


        # linear_loss = (1 - self.discount) * torch.mean(init_nu)
        non_linear_loss_mu_r = torch.logsumexp(imperfect_adv_mu_r)
        non_linear_loss_mu_nu = self.non_expert_regularization * torch.logsumexp(imperfect_adv_mu_nu / self.non_expert_regularization)
        nu_mu_loss = non_linear_loss_mu_r + non_linear_loss_mu_nu #+linear_loss

        # weighted BC
        weight_sa = torch.math.exp((imperfect_adv_mu_nu - torch.max(imperfect_adv_mu_nu)) / self.non_expert_regularization)
        weight_sa = weight_sa.unsqueeze(1)
        weight_sa = weight_sa / torch.mean(weight_sa)
        weight_ss1 = torch.math.exp(imperfect_adv_mu_r - torch.max(imperfect_adv_mu_r))
        weight_ss1 = weight_ss1.unsqueeze(1)
        weight_ss1 = weight_ss1 / torch.mean(weight_ss1)
            
        pi_loss = - torch.mean(
            weight_sa.detach() * self.actor.log_prob(imperfect_states, imperfect_actions))

        # gradient penalty for nu

        unif_rand2 = torch.random(expert_states.shape[0], 1)
        nu_inter = unif_rand2 * expert_states + (1 - unif_rand2) * imperfect_states
        nu_next_inter = unif_rand2 * expert_next_states + (1 - unif_rand2) * imperfect_next_states

        nu_inter = torch.cat([imperfect_states, nu_inter, nu_next_inter], 0)
        mu_inter = unif_rand2 * expert_inputs + (1 - unif_rand2) * imperfect_inputs
        mu_inter = torch.cat([imperfect_inputs, mu_inter], 0)

        with torch.no_grad():
            nu_output = self.nu(nu_inter)
            mu_output = self.mu(mu_inter)

        nu_mixed_grad = torch_grad(outputs=nu_output, inputs=nu_inter,
                               grad_outputs=torch.ones(nu_output.size()).cuda() if self.use_cuda else torch.ones(
                               nu_output.size()),
                               create_graph=True, retain_graph=True)[0] + EPS
        mu_mixed_grad = torch_grad(outputs=mu_output, inputs=interpolated,
                               grad_outputs=torch.ones(mu_output.size()).cuda() if self.use_cuda else torch.ones(
                               mu_output.size()),
                               create_graph=True, retain_graph=True)[0] + EPS

        nu_grad_penalty = torch.sqrt(torch.sum(nu_mixed_grad ** 2, dim=-1))
        mu_grad_penalty = torch.sqrt(torch.sum(mu_mixed_grad ** 2, dim=-1))
       
        nu_mu_loss = nu_mu_loss + self.grad_reg_coeffs[1] * (nu_grad_penalty + mu_grad_penalty)
        
        nu_mu_loss.backward()
        pi_loss.backward()
        cost_loss.backward()
        
        self.cost_optimizer.step()
        self.critic_optimizer.step()
        self.actor_optimizer.step()

        info_dict = {
            'cost_loss': cost_loss,
            'nu_mu_loss': nu_mu_loss,
            'actor_loss': pi_loss,
            'expert_nu': torch.mean(expert_nu),
            'imperfect_nu': torch.mean(imperfect_nu),
            # 'init_nu': torch.mean(init_nu),
            'imperfect_adv': torch.mean(imperfect_adv_mu_nu),
        }

        return info_dict



    def step(self, observation, deterministic=False):
        pred_action, _ = self.actor(observation, deterministic=False)
        return pred_action

    
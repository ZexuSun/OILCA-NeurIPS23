from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
import numpy as np
#from causal_world.evaluation.evaluation import EvaluationPipeline
#import causal_world.evaluation.protocols as protocols

# log_relative_path = './pushing_policy_tutorial_1'


def _make_env(rank):

    def _init():
        task = generate_task(task_generator_id="pushing",
                          variables_space='space_a')
        env = CausalWorld(task=task, enable_visualization=False, seed=rank)
        
        return env

    set_global_seeds(0)
    return _init


def train_policy():
    """
    ppo_config = {
        "gamma": 0.9988,
        "n_steps": 200,
        "ent_coef": 0,
        "learning_rate": 0.001,
        "vf_coef": 0.99,
        "max_grad_norm": 0.1,
        "lam": 0.95,
        "nminibatches": 5,
        "noptepochs": 100,
        "cliprange": 0.2
    }
    """
    ppo_config = {
        "gamma": 0.9995,
        "n_steps": 5000,
        "ent_coef": 0,
        "learning_rate": 0.00025,
        "vf_coef": 0.5,
        "max_grad_norm": 10,
        "nminibatches": 1000,
        "noptepochs": 4,
        "cliprange": 0.2
    }
    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[128, 128])
    env = SubprocVecEnv([_make_env(rank=i) for i in range(3)])
    #env = SubprocVecEnv([env])
    model = PPO2(MlpPolicy,
                 env,
                 _init_setup_model=True,
                 policy_kwargs=policy_kwargs,
                 verbose=1,
                 **ppo_config)
    model.learn(total_timesteps=5000000,
                tb_log_name="ppo2",
                reset_num_timesteps=False)


    env.close()
    return model

def online_test(policy):
    for rank in range(3):
        task = generate_task(task_generator_id='pushing',
                            variables_space='space_a')
        env = CausalWorld(task=task, enable_visualization=False, seed=rank)
        obs = env.reset()
        top_rewards = []
        for _ in range(500):
            episode_reward = 0
            fractional_success = 0
            for i in range(1000):
                obs, reward, done, info = env.step(model.predict(obs)[0])
                fractional_success += info['fractional_success']
                episode_reward += reward
                if done == True:
                    print('step', i+1)
                    break
            print('episode_reward', episode_reward)
            print('mean fractional_success', episode_reward/(i+1))

            if episode_reward > 0:
                top_rewards.append(episode_reward)
            obs = env.reset()
        print(len(top_rewards), 'expert number')
        env.close()


if __name__ == '__main__':
    model = train_policy()
    online_test(model)
    # evaluate_trained_policy()
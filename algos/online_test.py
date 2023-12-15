from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import generate_task
import numpy as np


def online_test(policy):
    task = generate_task(task_generator_id='pushing',
                          variables_space='space_b')
    env = CausalWorld(task=task, enable_visualization=False)
    obs = env.reset()
    for _ in range(500):
        epsoide_reward = 0
        for i in range(1000):
            action, _ =  policy(obs)
            obs, reward, done, info = env.step()

        if epsoide_reward > 0:
            top_rewards.append(epsoide_reward)
        obs = env.reset()
    print(len(top_rewards), 'expert number')
    env.close()

if __name__ == '__main__':
    online_test(policy)
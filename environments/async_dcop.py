import random
from functions import *
import matplotlib.pyplot as plt

from plot_functions.plot_functions import *
import numpy as np

from globals import *


class Message:
    def __init__(self, sender, receiver, content, t_start, t_end):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.t_start = t_start
        self.t_end = t_end


class AsyncMailer:
    def __init__(self, agents, max_steps):
        self.agents = agents
        self.max_steps = max_steps
        self.mailbox = {i: {agent.name: [] for agent in agents} for i in range(max_steps)}

    def send_messages(self, sender, messages, t_start):
        for receiver, content in messages.items():
            # t_end = t_start + 0
            t_end = t_start + random.randint(0, 10)
            async_message = Message(sender, receiver, content, t_start, t_end)
            if t_end < self.max_steps:
                self.mailbox[async_message.t_end][receiver].append(async_message)
                self.mailbox[async_message.t_end][receiver].sort(key=lambda x: x.t_start)


class AsyncAgent:
    def __init__(self, num, domain_size):
        self.num = num
        self.name = f'agent_{self.num}'
        self.d_size = domain_size
        self.domain = np.arange(self.d_size)
        self.constr_dict = {}
        self.nei_list = None
        self.action = None
        self.mailbox = {}
        self.step_count = 0

    def create_nei_list(self):
        self.nei_list = list(self.constr_dict.keys())

    def sample_action(self):
        self.action = random.choice(self.domain)
        return self.action

    def compute(self, obs):
        self.step_count = obs['step_count']
        self.mailbox[self.step_count] = obs
        choice = random.choice(self.domain)
        self.action = choice
        messages = {nei: self.action for nei in self.nei_list}
        return choice, messages




class AsyncDCOP:
    def __init__(self, max_steps=120, n_agents=None, domain_size=None, constraints_type=None):
        self.max_steps = max_steps
        self.n_agents = n_agents
        self.domain_size = domain_size
        self.constraints_type = constraints_type
        self.agents, self.agents_dict, self.agent_initial_action = None, None, None
        self.step_count = None
        self.mailer = None

        # for rendering
        self.fig, self.ax = plt.subplots(2, 2, figsize=(12, 8))
        self.h_real_rewards = None

    def create_new_problem(self):
        # create agents
        self.agents, self.agents_dict, self.agent_initial_action = [], {}, {}
        for i_agent in range(self.n_agents):
            agent = AsyncAgent(i_agent, self.domain_size)
            self.agents.append(agent)
            self.agents_dict[agent.name] = agent
            self.agent_initial_action[agent.name] = agent.sample_action()

        # create constraints
        if self.constraints_type == 'clique':
            for agent1, agent2 in combinations(self.agents, 2):
                # print(f'{agent1.name} - {agent2.name}')
                constr_for_agent_1 = np.random.randint(0, 100, (agent1.d_size, agent2.d_size))
                agent1.constr_dict[agent2.name] = constr_for_agent_1
                constr_for_agent_2 = constr_for_agent_1.T
                agent2.constr_dict[agent1.name] = constr_for_agent_2

        # create nei_list
        _ = [agent.create_nei_list() for agent in self.agents]

    def create_obs(self):
        # get the relevant info from mailer
        observation = {
            agent.name: {
                'choice': agent.action,
                'step_count': self.step_count,
                'messages': self.mailer.mailbox[self.step_count][agent.name]
            }
            for agent in self.agents}
        return observation

    def reset(self):
        # resets existing created problem
        self.step_count = 0

        # initial action
        for agent in self.agents:
            agent.action = self.agent_initial_action[agent.name]

        # reset mailer
        self.mailer = AsyncMailer(self.agents, self.max_steps)
        observation = self.create_obs()
        info = {}

        # for rendering
        self.h_real_rewards = []

        return observation, info

    def sample_actions(self, observation):
        actions = {}
        for agent in self.agents:
            obs = observation[agent.name]
            actions[agent.name] = agent.compute(obs)
        return actions

    def calc_real_rewards(self):
        rewards = {}
        for agent in self.agents:
            agent_reward = 0
            for nei_name in agent.constr_dict.keys():
                nei_agent = self.agents_dict[nei_name]
                agent_reward += agent.constr_dict[nei_name][agent.action, nei_agent.action]
                # if agent.name == 'agent_0' and nei_name == 'agent_1':
                #     print(f'[{self.step_count}] real_reward: {agent.constr_dict[nei_name][agent.action, nei_agent.action]}')
            rewards[agent.name] = agent_reward
        return rewards

    def step(self, actions):

        # execute actions
        for agent in self.agents:
            choice, messages = actions[agent.name]
            agent.action = choice
            self.mailer.send_messages(agent.name, messages, self.step_count)

        observation = self.create_obs()
        rewards = {}
        terminated = True if self.step_count >= self.max_steps else False
        truncated = False
        info = {}

        # for rendering
        real_rewards = self.calc_real_rewards()
        self.h_real_rewards.append(sum(real_rewards.values()))

        self.step_count += 1
        return observation, rewards, terminated, truncated, info

    def close(self):
        pass

    def render(self, info, plot_every=20):
        if self.step_count % plot_every == 0:

            info.update({
                'max_steps': self.max_steps,
                'real_rewards': self.h_real_rewards,
            })
            plot_rewards(self.ax[0, 0], info)

            plt.pause(0.001)


def main():
    random_seed_bool = False
    # random_seed_bool = True
    set_seed(random_seed_bool)

    n_agents = 10
    domain_size = 5
    constraints_type = 'clique'

    n_problems = 1

    env = AsyncDCOP(
        max_steps=120,
        n_agents=n_agents, domain_size=domain_size, constraints_type=constraints_type
    )

    for i_problem in range(n_problems):
        env.create_new_problem()
        observation, info = env.reset()

        for i in range(env.max_steps):

            # choose actions
            action = env.sample_actions(observation)

            # execute actions
            new_observation, rewards, terminated, truncated, info = env.step(action)

            # after actions
            if terminated or truncated:
                observation, info = env.reset()
            else:
                observation = new_observation

            # render
            env.render()

            # stats
            pass

    plt.show()
    env.close()


if __name__ == '__main__':
    main()


# import gymnasium as gym
# env = gym.make("LunarLander-v2", render_mode="human")
# observation, info = env.reset(seed=42)
# for _ in range(1000):
#    action = env.action_space.sample()  # this is where you would insert your policy
#    observation, reward, terminated, truncated, info = env.step(action)
#
#    if terminated or truncated:
#       observation, info = env.reset()
# env.close()
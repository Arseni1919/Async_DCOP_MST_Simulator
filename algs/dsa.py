from globals import *
from functions import *
from environments.async_dcop import AsyncDCOP


class DCOPAgent:
    def __init__(self, agent):
        self.num = agent.num
        self.name = agent.name
        self.d_size = agent.d_size
        self.domain = agent.domain
        self.constr_dict = agent.constr_dict
        self.nei_list = agent.nei_list
        self.action = agent.action
        # self.mailbox = agent.mailbox
        self.beliefs = {}
        self.last_actions = {nei: 0 for nei in self.nei_list}
        self.step_count = 0

    def update_beliefs(self, messages):
        for message in messages:
            self.beliefs[message.sender] = message.content
            self.last_actions[message.sender] = message.content

    def create_actions_dict(self):
        actions_dict = {}
        for action in self.domain:
            action_value = 0
            for nei in self.nei_list:
                action_value += self.constr_dict[nei][action, self.beliefs[nei]]
            actions_dict[action] = action_value
        return actions_dict

    def pick_dsa_action(self):
        actions_dict = self.create_actions_dict()
        min_action = min(actions_dict, key=actions_dict.get)
        min_value = actions_dict[min_action]
        curr_value = actions_dict[self.action]
        chosen_action = self.action
        if min_value < curr_value:
            if random.random() < 0.8:
                chosen_action = min_action
        return chosen_action

    def all_arrived(self):
        for nei in self.nei_list:
            if nei not in self.beliefs:
                return False
        return True

    def compute(self, obs):
        # self.mailbox[self.step_count] = obs
        self.step_count = obs['step_count']
        self.update_beliefs(obs['messages'])
        messages = {}
        if self.step_count == 0:
            messages = {nei: self.action for nei in self.nei_list}

        if self.all_arrived():
            # choice = random.choice(self.domain)
            choice = self.pick_dsa_action()
            self.action = choice
            self.beliefs = {}
            messages = {nei: self.action for nei in self.nei_list}

        return self.action, messages

    def calc_obs_reward(self):
        obs_reward = 0
        for nei_name in self.nei_list:
            nei_action = self.last_actions[nei_name]
            obs_reward += self.constr_dict[nei_name][self.action, nei_action]
            # if self.name == 'agent_0' and nei_name == 'agent_1':
            #     print(f'---\n[{self.step_count}] obs_reward: {self.constr_dict[nei_name][self.action, nei_action]}')
        return obs_reward


class AlgDSA:
    def __init__(self):
        self.agents = None
        self.agents_dict = None
        self.name = 'dsa'

        # STATS
        self.h_obs_rewards = []

    def reset(self, agents):
        self.agents = []
        self.agents_dict = {}
        for agent in agents:
            dsa_agent = DCOPAgent(agent)
            self.agents.append(dsa_agent)
            self.agents_dict[dsa_agent.name] = dsa_agent

    def calc_actions(self, observations):
        actions = {}
        for agent in self.agents:
            agent_obs = observations[agent.name]
            actions[agent.name] = agent.compute(agent_obs)

        # STATS
        obs_rewards = {agent.name: agent.calc_obs_reward() for agent in self.agents}
        self.h_obs_rewards.append(sum(obs_rewards.values()))
        return actions


def main():
    # random_seed_bool = False
    random_seed_bool = True
    set_seed(random_seed_bool)

    n_agents = 10
    domain_size = 5
    constraints_type = 'clique'

    n_problems = 1

    env = AsyncDCOP(
        max_steps=120,
        n_agents=n_agents, domain_size=domain_size, constraints_type=constraints_type
    )
    alg = AlgDSA()

    for i_problem in range(n_problems):
        env.create_new_problem()
        observation, info = env.reset()
        alg.reset(env.agents)

        for i in range(env.max_steps):

            # choose actions
            actions = alg.calc_actions(observation)

            # execute actions
            new_observation, rewards, terminated, truncated, info = env.step(actions)

            # after actions
            if terminated or truncated:
                observation, info = env.reset()
            else:
                observation = new_observation

            # render
            info = {
                'obs_rewards': alg.h_obs_rewards
            }
            env.render(info)

            # stats
            pass

    plt.show()
    env.close()


if __name__ == '__main__':
    main()

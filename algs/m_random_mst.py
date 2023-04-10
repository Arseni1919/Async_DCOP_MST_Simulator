from globals import *
from algs.test_mst_alg import test_mst_alg
from algs.alg_objects import *


class RandomMstAlgAgent(AlgAgent):
    def __init__(self, sim_agent):
        super(RandomMstAlgAgent, self).__init__(sim_agent)


class RandomMstAlg:
    def __init__(self):
        self.agents, self.agents_dict = None, None
        self.name = 'random'

    def create_entities(self, sim_agents, sim_targets):
        self.agents, self.agents_dict = [], {}
        for sim_agent in sim_agents:
            new_agent = RandomMstAlgAgent(sim_agent)
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent

    def reset(self, sim_agents, sim_targets):
        self.create_entities(sim_agents, sim_targets)

    def get_actions(self, observations):
        actions = {agent.name: {
            'move': random.randint(0, 4),
            'send': []
        }
            for agent in self.agents
        }
        return actions

    def get_info(self):
        info = {}
        return info


def main():
    alg = RandomMstAlg()
    test_mst_alg(alg)


if __name__ == '__main__':
    main()

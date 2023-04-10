from globals import *
from algs.test_mst_alg import test_mst_alg
from algs.alg_functions import *
from algs.alg_objects import *


class DsaMstAlgAgent(AlgAgent):
    def __init__(self, sim_agent):
        super(DsaMstAlgAgent, self).__init__(sim_agent)

    def get_move_order(self):
        if self.is_moving:
            return -1  # wait
        possible_actions = list(self.pos.actions_dict.keys())
        possible_next_pos = list(self.pos.actions_dict.values())
        targets = [AttributeDict(target) for target in self.nei_targets]
        if len(targets) == 0:
            return random.choice(possible_actions)

        new_pos = select_pos(self, targets, possible_next_pos)
        action = list(filter(lambda x: self.pos.actions_dict[x].xy_name == new_pos.xy_name, self.pos.actions_dict))[0]
        if get_dsa_mst_replacement_decision(self, new_pos, targets):
            return action
        return random.choice(possible_actions)

    def get_send_order(self):
        return []

    def process(self, observation):
        self.observe(observation)
        move_order = self.get_move_order()
        send_order = self.get_send_order()
        return move_order, send_order


class DsaMstAlg:
    def __init__(self):
        self.name = 'DSA_MST'
        self.agents, self.agents_dict = None, None

    def create_entities(self, sim_agents, sim_targets):
        self.agents, self.agents_dict = [], {}
        for sim_agent in sim_agents:
            new_agent = DsaMstAlgAgent(sim_agent)
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent

    def reset(self, sim_agents, sim_targets):
        self.create_entities(sim_agents, sim_targets)

    def get_actions(self, observations):
        actions = {}
        for agent in self.agents:
            move_order, send_order = agent.process(observations[agent.name])
            actions[agent.name] = {'move': move_order, 'send': send_order}
        return actions

    def get_info(self):
        info = {}
        return info


def main():
    alg = DsaMstAlg()
    # test_mst_alg(alg, to_render=False)
    # test_mst_alg(alg, to_render=True, plot_every=10)
    test_mst_alg(
        alg,
        n_agents=30,
        n_targets=10,
        to_render=True,
        plot_every=5
    )


if __name__ == '__main__':
    main()

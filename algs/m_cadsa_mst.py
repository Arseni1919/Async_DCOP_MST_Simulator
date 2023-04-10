from globals import *
from algs.test_mst_alg import test_mst_alg
from algs.alg_functions import *
from algs.alg_objects import *


class CaDsaMstAlgAgent(AlgAgent):
    def __init__(self, sim_agent):
        super(CaDsaMstAlgAgent, self).__init__(sim_agent)
        self.to_calc_new_pos = True
        self.next_possible_pos, self.next_possible_action = None, None
        self.beliefs = {}

    def update_beliefs(self):
        """
        income messages -> [(from, s_time, content), ...]
        self.mailbox[self.step_count] = observation.new_messages
        """
        for nei in self.nei_agents:
            nei_name = nei['name']
            # create belief if this is new agent
            if nei_name not in self.beliefs:
                self.beliefs[nei_name] = {
                    'last_s_time': 0,
                    'sync_count': None,
                    'next_possible_pos': None}

        new_messages = self.mailbox[self.step_count]
        for from_a_name, s_time, content in new_messages:
            # if old message -> ignore
            if s_time >= self.beliefs[from_a_name]['last_s_time']:
                self.beliefs['last_s_time'] = s_time
                self.beliefs['sync_count'] = content['sync_count']
                self.beliefs['next_possible_pos'] = content['next_possible_pos']

    def decide_next_possible_move(self):
        self.to_calc_new_pos = False
        self.next_possible_pos, self.next_possible_action = None, None
        possible_actions = list(self.pos.actions_dict.keys())
        possible_next_pos = list(self.pos.actions_dict.values())
        targets = [AttributeDict(target) for target in self.nei_targets]
        if len(targets) == 0:
            self.next_possible_action = random.choice(possible_actions)
            self.next_possible_pos = self.pos.actions_dict[self.next_possible_action]
        else:
            new_pos = select_pos(self, targets, possible_next_pos)
            new_action = list(filter(lambda x: self.pos.actions_dict[x].xy_name == new_pos.xy_name, self.pos.actions_dict))[0]
            if get_dsa_mst_replacement_decision(self, new_pos, targets):
                self.next_possible_action = new_action
                self.next_possible_pos = new_pos
            else:
                self.next_possible_action = random.choice(possible_actions)
                self.next_possible_pos = self.pos.actions_dict[self.next_possible_action]

    def get_move_order(self):
        if self.is_moving:
            return -1  # wait

        # decide what next possible action to send
        if self.to_calc_new_pos:
            self.decide_next_possible_move()

        # wait for all neighbours
        for nei in self.nei_agents:
            if self.beliefs[nei['name']]['sync_count'] != self.sync_count:
                return -1
        self.to_calc_new_pos = True

        # if there is a possible collision -> stay on place
        for nei in self.nei_agents:
            nei_next_possible_pos = self.beliefs[nei['name']]['next_possible_pos']
            if nei_next_possible_pos.xy_name == self.next_possible_pos.xy_name:
                return 0
        return self.next_possible_action

    def get_send_order(self):
        """
        SEND ORDER: message -> [(from, to, s_time, content), ...]
        """
        messages = []
        for nei in self.nei_agents:
            content = {
                'sync_count': self.sync_count,
                'next_possible_pos': self.next_possible_pos,
            }
            new_message = (self.name, nei['name'], self.step_count, content)
            messages.append(new_message)
        return messages

    def process(self, observation):
        self.observe(observation)
        self.update_beliefs()
        move_order = self.get_move_order()
        send_order = self.get_send_order()
        return move_order, send_order


class CaDsaMstAlg:
    def __init__(self):
        self.name = 'CADSA'
        self.agents, self.agents_dict = None, None
        self.sync_count = None

    def create_entities(self, sim_agents, sim_targets):
        self.sync_count = 0
        self.agents, self.agents_dict = [], {}
        for sim_agent in sim_agents:
            new_agent = CaDsaMstAlgAgent(sim_agent)
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
    alg = CaDsaMstAlg()
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

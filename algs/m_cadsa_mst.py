from globals import *
from algs.test_mst_alg import test_mst_alg
from algs.alg_functions import *
from algs.alg_objects import *


class CaDsaMstAlgConductor:
    def __init__(self, agents):
        self.agents = agents
        self.iteration = 0
        self.agents_finished_move_dict = {self.iteration: {agent.name: False for agent in self.agents}}
        self.agents_ready_move_dict = {self.iteration: {agent.name: False for agent in self.agents}}
        self.started_to_move_dict = {self.iteration: {agent.name: False for agent in self.agents}}

    def advance(self):
        self.iteration += 1
        self.agents_finished_move_dict[self.iteration] = {agent.name: False for agent in self.agents}
        self.agents_ready_move_dict[self.iteration] = {agent.name: False for agent in self.agents}
        self.started_to_move_dict[self.iteration] = {agent.name: False for agent in self.agents}

    def finished_to_move(self, agent):
        self.agents_finished_move_dict[self.iteration][agent.name] = True

    def ready_to_move(self, agent):
        self.agents_ready_move_dict[self.iteration][agent.name] = True

    def started_to_move(self, agent):
        self.started_to_move_dict[self.iteration][agent.name] = True

    def finished_to_move_all(self):
        return all(self.agents_finished_move_dict[self.iteration].values())

    def ready_to_move_all(self):
        return all(self.agents_ready_move_dict[self.iteration].values())

    def started_to_move_all(self):
        return all(self.started_to_move_dict[self.iteration].values())


class CaDsaMstAlgAgent(AlgAgent):
    def __init__(self, sim_agent, conductor=None):
        super(CaDsaMstAlgAgent, self).__init__(sim_agent)
        self.to_calc_new_pos = True
        self.conductor = conductor
        self.next_possible_pos, self.next_possible_action = None, None
        self.beliefs = {}
        self.next_pos_ready = False
        self.sync_time = 0

    def reset_beliefs(self):
        self.beliefs[self.conductor.iteration] = {}
        for agent in self.conductor.agents:
            # create belief if this is new agent
            self.beliefs[self.conductor.iteration][agent.name] = {'next_possible_pos': None}

    def update_beliefs(self):
        """
        income messages -> [(from, s_time, content), ...]
        self.mailbox[self.step_count] = observation.new_messages
        """
        if self.conductor.iteration not in self.beliefs:
            return
        new_messages = self.mailbox[self.step_count]
        for from_a_name, s_time, content in new_messages:
            # if old message -> ignore
            if s_time > self.sync_time:
                self.beliefs[self.conductor.iteration][from_a_name]['next_possible_pos'] = content['next_possible_pos']

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
            new_action = \
            list(filter(lambda x: self.pos.actions_dict[x].xy_name == new_pos.xy_name, self.pos.actions_dict))[0]
            if get_dsa_mst_replacement_decision(self, new_pos, targets):
                self.next_possible_action = new_action
                self.next_possible_pos = new_pos
            else:
                self.next_possible_action = random.choice(possible_actions)
                self.next_possible_pos = self.pos.actions_dict[self.next_possible_action]

    def action_without_collisions(self):
        # if there is a possible collision -> stay on place
        for nei in self.nei_agents:
            nei_name = nei['name']
            nei_next_possible_pos = self.beliefs[self.conductor.iteration][nei_name]['next_possible_pos']
            if nei_next_possible_pos is None:
                return -1
            if self.next_possible_pos.xy_name == nei_next_possible_pos.xy_name:
                return 0
            if self.next_possible_pos.xy_name == nei['pos'].xy_name:
                return 0
        return self.next_possible_action

    def get_move_order(self):
        if self.is_moving:
            return -1  # wait

        self.conductor.finished_to_move(self)

        # wait for all neighbours
        if not self.conductor.finished_to_move_all():
            return -1

        if not self.next_pos_ready:
            # decide what next possible action to send
            self.decide_next_possible_move()
            self.reset_beliefs()
            self.sync_time = self.step_count
            self.next_pos_ready = True

        # if there is a possible collision -> stay on place
        next_action = self.action_without_collisions()

        if next_action != -1:
            self.conductor.ready_to_move(self)

        if not self.conductor.ready_to_move_all():
            return -1

        self.next_pos_ready = False
        self.conductor.started_to_move(self)
        return next_action

    def get_send_order(self):
        """
        SEND ORDER: message -> [(from, to, s_time, content), ...]
        """
        messages = []
        for nei in self.nei_agents:
            content = {
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
        self.conductor = None

    def create_entities(self, sim_agents, sim_targets):
        self.agents, self.agents_dict = [], {}
        for sim_agent in sim_agents:
            new_agent = CaDsaMstAlgAgent(sim_agent)
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent
        self.conductor = CaDsaMstAlgConductor(self.agents)
        for agent in self.agents:
            agent.conductor = self.conductor
            agent.reset_beliefs()

    def reset(self, sim_agents, sim_targets):
        self.create_entities(sim_agents, sim_targets)

    def get_actions(self, observations):
        actions = {}
        for agent in self.agents:
            move_order, send_order = agent.process(observations[agent.name])
            actions[agent.name] = {'move': move_order, 'send': send_order}
        if self.conductor.started_to_move_all():
            self.conductor.advance()
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
        plot_every=10,
        random_seed_bool=True,
        seed=572
    )


if __name__ == '__main__':
    main()

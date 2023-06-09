from globals import *
from algs.test_mst_alg import test_mst_alg
from algs.alg_functions import *
from algs.alg_objects import *
from functions import *


class CaDsaMstAlgAgent(AlgAgent):
    def __init__(self, sim_agent):
        super(CaDsaMstAlgAgent, self).__init__(sim_agent)
        self.next_possible_pos, self.next_possible_action = None, None
        self.next_action = -1
        self.beliefs = {}

        # states
        """
        first
        v
        f_move -> exchange -> f_exchange -> plan -> f_plan -> ca_correction -> f_ca_correction -> move
        ^                                                                                            v
        <-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-
        """
        self.sync_time = 0
        self.state_counter = 0
        self.state = 'first'

    def reset_beliefs(self):
        self.beliefs = {}
        for agent in self.all_agents:
            # create belief if this is new agent
            self.beliefs[agent['name']] = {'next_possible_pos': None, 'state': ''}

    def update_beliefs(self):
        """
        income messages -> [(from, s_time, content), ...]
        self.mailbox[self.step_count] = observation.new_messages
        """
        new_messages = self.mailbox[self.step_count]
        for from_a_name, s_time, content in new_messages:
            # if old message -> ignore
            if s_time > self.sync_time:
                if self.state == content['state']:
                    self.beliefs[from_a_name]['state'] = content['state']
                self.beliefs[from_a_name]['next_possible_pos'] = content['next_possible_pos']

    def decide_next_possible_move(self):
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
        if self.next_possible_pos is None or self.next_possible_action is None:
            raise RuntimeError()

    def action_without_collisions(self):
        # if there is a possible collision -> stay on place
        for nei in self.nei_agents:
            nei_name = nei['name']
            nei_next_possible_pos = self.beliefs[nei_name]['next_possible_pos']
            if nei_next_possible_pos is None:
                raise RuntimeError()
            if self.next_possible_pos.xy_name == nei_next_possible_pos.xy_name:
                return 0
            if self.next_possible_pos.xy_name == nei['pos'].xy_name:
                return 0
        return self.next_possible_action

    def get_send_order(self, show_next_pos=True):
        """
        SEND ORDER: message -> [(from, to, s_time, content), ...]
        """
        if not show_next_pos:
            self.next_pos = None
            self.next_action = None
            self.next_possible_action = None
            self.next_possible_pos = None
        messages = []
        for agent in self.all_agents:
            content = {
                'state': self.state,
                'next_possible_pos': self.next_possible_pos,
            }
            new_message = (self.name, agent['name'], self.step_count, content)
            messages.append(new_message)
        return messages

    def all_states_aligned(self):
        for agent_name, belief in self.beliefs.items():
            state = belief['state']
            if state != self.state:
                return False
        return True

    def received_all_next_possible_pos(self):
        for agent in self.nei_agents:
            nei_name = agent['name']
            nei_next_possible_pos = self.beliefs[nei_name]['next_possible_pos']
            if nei_next_possible_pos is None:
                return False
        return True

    def state_first(self):
        self.reset_beliefs()
        self.state = 'f_move'
        move_order = -1
        send_order = self.get_send_order(show_next_pos=False)
        return move_order, send_order

    def state_move(self):
        if self.state != 'move':
            raise RuntimeError()

        if self.is_moving:
            move_order, send_order = -1, []
            self.sync_time = self.step_count
            return move_order, send_order  # wait
        self.state_counter += 1
        self.state = 'f_move'
        move_order = -1
        send_order = self.get_send_order(show_next_pos=False)
        return move_order, send_order

    def state_f_move(self):
        move_order = -1
        send_order = self.get_send_order(show_next_pos=False)
        if self.all_states_aligned():
            self.reset_beliefs()
            self.state = 'plan'
        return move_order, send_order

    def state_plan(self):
        self.decide_next_possible_move()
        self.state = 'f_plan'
        move_order = -1
        send_order = self.get_send_order()
        return move_order, send_order

    def state_f_plan(self):
        move_order = -1
        send_order = self.get_send_order()
        if self.all_states_aligned():
            self.state = 'ca_correction'
        return move_order, send_order

    def state_ca_correction(self):
        move_order = -1
        send_order = self.get_send_order()
        if self.received_all_next_possible_pos():
            self.next_action = self.action_without_collisions()
            self.state = 'f_ca_correction'
        return move_order, send_order

    def state_f_ca_correction(self):
        move_order = -1
        send_order = self.get_send_order()
        if self.all_states_aligned():
            self.state = 'move'
            return self.next_action, send_order
        return move_order, send_order

    def process(self, observation):
        self.observe(observation)
        self.update_beliefs()

        if self.state == 'first':
            move_order, send_order = self.state_first()

        elif self.state == 'move':
            move_order, send_order = self.state_move()

        elif self.state == 'f_move':
            move_order, send_order = self.state_f_move()

        elif self.state == 'plan':
            move_order, send_order = self.state_plan()

        elif self.state == 'f_plan':
            move_order, send_order = self.state_f_plan()

        elif self.state == 'ca_correction':
            move_order, send_order = self.state_ca_correction()

        elif self.state == 'f_ca_correction':
            move_order, send_order = self.state_f_ca_correction()

        else:
            raise RuntimeError('unknown state')

        return move_order, send_order


class CaDsaMstAlg:
    def __init__(self):
        self.name = 'CADSA'
        self.agents, self.agents_dict = None, None

    def create_entities(self, sim_agents, sim_targets):
        self.agents, self.agents_dict = [], {}
        for sim_agent in sim_agents:
            new_agent = CaDsaMstAlgAgent(sim_agent)
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent

    def reset(self, sim_agents, sim_targets):
        self.create_entities(sim_agents, sim_targets)

    def get_actions(self, observations):
        actions = {}
        # state_counter
        print(f' ------------------------------ step: {observations["step_count"]} ------------------------------ ')
        for agent in self.agents:
            move_order, send_order = agent.process(observations[agent.name])
            actions[agent.name] = {'move': move_order, 'send': send_order}
            print(f"{agent.name}'s state counter: {agent.state_counter}, state: {agent.state}")
        return actions

    def get_info(self):
        info = {}
        return info


def main():
    set_seed(random_seed_bool=False, i_seed=902)
    # set_seed(random_seed_bool=True)
    alg = CaDsaMstAlg()
    # test_mst_alg(alg, to_render=False)
    # test_mst_alg(alg, to_render=True, plot_every=10)
    # set_seed(True, 353)
    test_mst_alg(
        alg,
        n_agents=30,
        n_targets=10,
        to_render=True,
        plot_every=50,
        n_problems=3,
        max_steps=500,
    )


if __name__ == '__main__':
    main()

from globals import *
from algs.test_mst_alg import test_mst_alg
from algs.alg_functions import *
from algs.alg_objects import *
from functions import *


class DssaAlgAgent(AlgAgent):
    def __init__(self, sim_agent):
        super(DssaAlgAgent, self).__init__(sim_agent)

        # for sync
        self.beliefs = {}
        self.state = 'first'
        self.sync_time = 0
        self.state_counter = 0
        self.next_possible_pos, self.next_possible_action = None, None

        # for DSSA
        self.need_to_talk = True
        self.tries = None
        self.max_tries = 20
        self.possible_next_pos_list = None

    def reset_beliefs(self):
        self.beliefs = {}
        for agent in self.all_agents:
            # create belief if this is new agent
            self.beliefs[agent['name']] = {
                'next_possible_pos': None,
                'state': '',
                'tries': None,
                'need_to_talk': False,
            }

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
                if self.state == 'plan':
                    self.beliefs[from_a_name]['tries'] = content['tries']
                    self.beliefs[from_a_name]['need_to_talk'] = content['need_to_talk']
                self.beliefs[from_a_name]['next_possible_pos'] = content['next_possible_pos']

    def all_states_aligned(self):
        for agent_name, belief in self.beliefs.items():
            state = belief['state']
            if state != self.state:
                return False
        return True

    def nei_tries_aligned(self):
        for nei in self.nei_agents:
            nei_name = nei['name']
            tries = self.beliefs[nei_name]['tries']
            if tries != self.tries:
                return False
        return True

    def get_send_order(self, show_next_pos=True):
        """
        SEND ORDER: message -> [(from, to, s_time, content), ...]
        """
        if not show_next_pos:
            self.next_pos = None
            self.next_possible_action = None
            self.next_possible_pos = None
        messages = []
        for agent in self.all_agents:
            content = {
                'state': self.state,
                'next_possible_pos': self.next_possible_pos,
                'tries': self.tries,
                'need_to_talk': self.need_to_talk
            }
            new_message = (self.name, agent['name'], self.step_count, content)
            messages.append(new_message)
        return messages

    def decide_next_possible_move(self, robot_pos_name_set=None):
        self.next_possible_pos, self.next_possible_action = None, None
        possible_actions = list(self.pos.actions_dict.keys())
        possible_next_pos = list(self.pos.actions_dict.values())
        targets = [AttributeDict(target) for target in self.nei_targets]
        if len(targets) == 0:
            self.next_possible_action = random.choice(possible_actions)
            self.next_possible_pos = self.pos.actions_dict[self.next_possible_action]
        else:
            new_pos = select_pos(self, targets, possible_next_pos, robot_pos_name_set=robot_pos_name_set)
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

    def there_is_a_collision(self):
        # if there is a possible collision -> stay on place
        for nei in self.nei_agents:
            nei_name = nei['name']
            nei_pos = nei['pos']
            nei_next_possible_pos = self.beliefs[nei_name]['next_possible_pos']
            # nei_need_to_talk = self.beliefs[nei_name]['need_to_talk']
            # if nei_need_to_talk:
            #     return True
            if nei_next_possible_pos is None:
                raise RuntimeError()
            if self.next_possible_pos.xy_name == nei_next_possible_pos.xy_name:
                return True
            if self.next_possible_pos.xy_name == nei_pos.xy_name:
                return True
        return False

    def state_first(self):
        self.reset_beliefs()
        self.state = 'first_plan'
        self.tries = 0
        move_order = -1
        send_order = self.get_send_order(show_next_pos=False)
        return move_order, send_order

    def state_first_plan(self):
        self.decide_next_possible_move()
        self.possible_next_pos_list = list(self.pos.actions_dict.values())
        self.state = 'f_first_plan'
        move_order = -1
        send_order = self.get_send_order()
        return move_order, send_order

    def state_f_first_plan(self):
        move_order = -1
        send_order = self.get_send_order()
        if self.all_states_aligned():
            self.tries = 0
            self.state = 'plan'
        return move_order, send_order

    def state_plan(self):
        # TODO:
        """
        DSSA:
        Exchange next possible positions until the solution without collisions will be found.
        Each time exclude the problematic position.
        """
        move_order = -1
        send_order = self.get_send_order()

        self.need_to_talk = self.there_is_a_collision()
        # if collision with other agents - we still need to talk
        if self.need_to_talk:
            self.tries += 1
            nei_targets = [AttributeDict(target) for target in self.nei_targets]
            if not get_dsa_mst_replacement_decision(self, self.next_possible_pos, nei_targets):
                # set different next position
                if self.next_possible_pos in self.possible_next_pos_list:
                    self.possible_next_pos_list.remove(self.next_possible_pos)
                robot_pos_name_set = [pos.xy_name for pos in self.possible_next_pos_list]
                self.decide_next_possible_move(robot_pos_name_set=robot_pos_name_set)
        if not self.need_to_talk:
            self.state = 'f_plan'
        if self.tries > self.max_tries:
            self.state = 'f_plan'
            self.next_possible_pos, self.next_possible_action = self.pos, 0
        return move_order, send_order

    def state_f_plan(self):
        move_order = -1
        send_order = self.get_send_order()
        if self.all_states_aligned():
            self.tries = 0
            self.state = 'move'
            return self.next_possible_action, send_order
        return move_order, send_order

    def state_move(self):
        if self.is_moving:
            move_order, send_order = -1, []
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
            self.sync_time = self.step_count
            self.state = 'first_plan'
        return move_order, send_order

    def process(self, observation):
        self.observe(observation)
        self.update_beliefs()

        if self.state == 'first':
            move_order, send_order = self.state_first()

        elif self.state == 'first_plan':
            move_order, send_order = self.state_first_plan()

        elif self.state == 'f_first_plan':
            move_order, send_order = self.state_f_first_plan()

        elif self.state == 'plan':
            move_order, send_order = self.state_plan()

        elif self.state == 'f_plan':
            move_order, send_order = self.state_f_plan()

        elif self.state == 'move':
            move_order, send_order = self.state_move()

        elif self.state == 'f_move':
            move_order, send_order = self.state_f_move()

        else:
            raise RuntimeError('unknown state')

        # move_order = self.get_move_order()
        # send_order = self.get_send_order()
        # move_order, send_order = -1, []
        return move_order, send_order


class DssaAlg:
    def __init__(self):
        self.name = 'DSSA'
        self.agents, self.agents_dict = None, None

    def create_entities(self, sim_agents, sim_targets):
        self.agents, self.agents_dict = [], {}
        for sim_agent in sim_agents:
            new_agent = DssaAlgAgent(sim_agent)
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

            if agent.state == 'plan':
                print(f"{agent.name}'s state counter: {agent.state_counter}, state: {agent.state}, tries: {agent.tries}, len: {len(agent.possible_next_pos_list)}")
            else:
                print(f"{agent.name}'s state counter: {agent.state_counter}, state: {agent.state}")

        return actions

    def get_info(self):
        info = {}
        return info


def main():
    set_seed(random_seed_bool=False, i_seed=595)
    # set_seed(random_seed_bool=True)
    alg = DssaAlg()
    # test_mst_alg(alg, to_render=False)
    # test_mst_alg(alg, to_render=True, plot_every=10)
    # set_seed(True, 353)
    test_mst_alg(
        alg,
        n_agents=70,
        n_targets=10,
        to_render=True,
        plot_every=10,
        n_problems=3,
        max_steps=120,
    )


if __name__ == '__main__':
    main()

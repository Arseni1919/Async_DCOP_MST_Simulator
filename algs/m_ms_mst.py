from globals import *
from algs.test_mst_alg import test_mst_alg
from algs.alg_functions import *
from algs.alg_objects import *
from functions import *


class MaxSumMstAlgAgent(AlgAgent):
    def __init__(self, sim_agent, with_breakdowns):
        super(MaxSumMstAlgAgent, self).__init__(sim_agent)
        self.beliefs = {}
        self.with_breakdowns = with_breakdowns

        # states
        self.sync_time = 0
        self.state_counter = 0
        self.state = 'first'

        # max-sum
        self.next_ms_pos, self.next_ms_action = None, None
        self.max_small_iterations = 10
        self.small_iter = 0

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

    def update_breakdowns(self):
        if len(self.col_agents_list) > 0:
            self.next_ms_action = 404
            self.next_ms_pos = self.pos

    def decide_next_ms_move(self):
        next_action_value_dict = {}
        nei_targets = [AttributeDict(t) for t in self.nei_targets]
        for next_action, next_pos in self.pos.actions_dict.items():
            next_value = 0
            for target in nei_targets:
                if distance_nodes(next_pos, target.pos) <= self.sr:
                    if self.name in target.fmr_nei:
                        next_value += self.cred
            next_action_value_dict[next_action] = next_value
        max_value = max(next_action_value_dict.values())
        max_actions = [k for k, v in next_action_value_dict.items() if v == max_value]
        self.next_ms_action = random.choice(max_actions)
        self.next_ms_pos = self.pos.actions_dict[self.next_ms_action]

    def get_send_order(self, show_next_pos=True):
        """
        SEND ORDER: message -> [(from, to, s_time, content), ...]
        """
        if not show_next_pos:
            self.next_pos = None
        messages = []
        for agent in self.all_agents:
            content = {
                'state': self.state,
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

    # ------------------------------------ states ------------------------------------ #

    def state_first(self):
        self.reset_beliefs()
        self.state = 'f_move'
        move_order = -1
        self.small_iter = 0
        send_order = self.get_send_order(show_next_pos=False)
        return move_order, send_order

    def state_f_move(self):
        move_order = -1
        send_order = self.get_send_order(show_next_pos=False)
        if self.all_states_aligned():
            self.reset_beliefs()
            self.sync_time = self.step_count
            self.state = 'plan'
        return move_order, send_order

    def state_plan(self):
        # self.decide_next_possible_move()
        self.decide_next_ms_move()
        self.state = 'f_plan'
        move_order = -1
        send_order = self.get_send_order()
        return move_order, send_order

    def state_f_plan(self):
        move_order = -1
        send_order = self.get_send_order()
        if self.with_breakdowns:
            self.update_breakdowns()
        if self.all_states_aligned():
            self.state = 'move'
            return self.next_ms_action, send_order
        return move_order, send_order

    def state_move(self):
        move_order = -1
        send_order = self.get_send_order(show_next_pos=False)
        if self.is_moving:

            return move_order, send_order  # wait
        self.state_counter += 1
        self.state = 'f_move'
        return move_order, send_order

    def process(self, observation):
        self.observe(observation)
        self.update_beliefs()

        if self.state == 'first':
            move_order, send_order = self.state_first()

        elif self.state == 'f_move':
            move_order, send_order = self.state_f_move()

        elif self.state == 'plan':
            move_order, send_order = self.state_plan()

        elif self.state == 'f_plan':
            move_order, send_order = self.state_f_plan()

        elif self.state == 'move':
            move_order, send_order = self.state_move()

        else:
            raise RuntimeError('unknown state')

        return move_order, send_order


class MaxSumMstAlg:
    def __init__(self, with_breakdowns=False):
        self.name = 'Max-Sum'
        self.agents, self.agents_dict = None, None
        self.sim_agents, self.sim_targets = None, None
        self.with_breakdowns = with_breakdowns

    def create_entities(self, sim_agents, sim_targets, *args):
        self.sim_agents, self.sim_targets = sim_agents, sim_targets
        self.agents, self.agents_dict = [], {}
        for sim_agent in sim_agents:
            new_agent = MaxSumMstAlgAgent(sim_agent, self.with_breakdowns)
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent

    def reset(self, sim_agents, sim_targets, *args):
        self.create_entities(sim_agents, sim_targets)

    def get_actions(self, observations):
        actions = {}
        # state_counter
        print(f' ------------------------------ step: {observations["step_count"]} ------------------------------ ')
        for agent in self.agents:
            move_order, send_order = agent.process(observations[agent.name])
            actions[agent.name] = {'move': move_order, 'send': send_order}
            # print(f"{agent.name}'s state counter: {agent.state_counter}, state: {agent.state}")
            # for target in self.sim_targets:
            #     print(f'{target.name}: {target.fmr_nei=}')
        return actions

    def get_info(self):
        info = {}
        return info


def main():
    set_seed(random_seed_bool=False, i_seed=191)
    # set_seed(random_seed_bool=True)

    # alg = MaxSumMstAlg(with_breakdowns=False)
    alg = MaxSumMstAlg(with_breakdowns=True)

    # test_mst_alg(alg, to_render=False)
    # test_mst_alg(alg, to_render=True, plot_every=10)
    # set_seed(True, 353)
    test_mst_alg(
        alg,
        n_agents=10,
        n_targets=10,
        to_render=True,
        plot_every=50,
        n_problems=3,
        max_steps=2000,
        with_fmr=True,
    )


if __name__ == '__main__':
    main()

from globals import *
from algs.test_mst_alg import test_mst_alg
from algs.alg_functions import *
from algs.alg_objects import *
from functions import *


def get_messages_for_all_agents(self, show_next_pos=False):
    """
    SEND ORDER: message -> [(from, to, s_time, content), ...]
    """
    if not show_next_pos:
        self.next_pos = None
    messages = []
    for agent in self.all_agents:
        content = {
            'state': self.state,
            'small_iter': self.small_iter
        }
        new_message = (self.name, agent['name'], self.step_count, content)
        messages.append(new_message)
    return messages


def reset_beliefs(all_agents, nei_pos_nodes=None):
    all_entities = []
    all_entities.extend(all_agents)
    if nei_pos_nodes is not None:
        all_entities.extend(nei_pos_nodes)
    beliefs = {}
    for entity in all_entities:
        # create belief if this is new agent
        beliefs[entity['name']] = {
            'state': '',
            'small_iter': None
        }
    return beliefs


def update_beliefs(mailbox, step_count, sync_time, self_state, self_small_iter, beliefs):
    """
    income messages -> [(from, s_time, content), ...]
    self.mailbox[self.step_count] = observation.new_messages
    """
    new_messages = mailbox[step_count]
    for from_a_name, s_time, content in new_messages:
        # if old message -> ignore
        if s_time > sync_time:
            state = content['state']
            small_iter = content['small_iter']
            if self_state == state and self_small_iter == small_iter:
                beliefs[from_a_name]['state'] = state
                beliefs[from_a_name]['small_iter'] = small_iter


class CamsAlgPosNode:
    def __init__(self, pos, max_small_iterations=10):
        self.pos = pos
        self.name = pos.xy_name
        self.step_count = None
        self.mailbox = {}
        self.beliefs = {}

        # states
        self.sync_time = 0
        self.state_counter = 0
        self.state = 'first'
        self.max_small_iterations = max_small_iterations
        self.small_iter = 0

        self.nei_agents = None
        self.all_agents = None
        # self.all_pos_nodes = None

    def is_with_nei(self):
        return len(self.nei_agents) > 0

    def reset_beliefs(self):
        self.beliefs = reset_beliefs(self.all_agents)

    def update_beliefs(self):
        update_beliefs(self.mailbox, self.step_count, self.sync_time, self.state, self.small_iter, self.beliefs)

    def observe(self, observation):
        observation = AttributeDict(observation)
        self.step_count = observation.step_count
        self.pos = observation.pos
        self.nei_agents = observation.nei_agents
        self.all_agents = observation.all_agents
        # self.all_pos_nodes = observation.all_pos_nodes
        self.mailbox[self.step_count] = observation.new_messages

    def get_regular_send_order(self):
        messages = []
        for agent in self.nei_agents:
            content = {
                'state': self.state,
                'small_iter': self.small_iter
            }
            new_message = (self.name, agent['name'], self.step_count, content)
            messages.append(new_message)
        return messages

    def get_plan_send_order(self):
        messages = self.get_regular_send_order()
        return messages

    def all_states_aligned(self):
        for agent in self.nei_agents:
            agent_name = agent['name']
            state = self.beliefs[agent_name]['state']
            small_iter = self.beliefs[agent_name]['small_iter']
            if state != self.state:  #  or small_iter != self.small_iter
                return False
        return True
    # ------------------------------------ states ------------------------------------ #

    def state_first(self):
        self.reset_beliefs()
        self.state = 'plan'
        self.small_iter = 0
        return []

    def state_plan(self):
        # TODO:
        if len(self.nei_agents) == 0:
            return []
        self.state = 'f_plan'
        self.small_iter += 1
        self.sync_time = self.step_count
        send_order = self.get_plan_send_order()
        return send_order

    def state_f_plan(self):
        if len(self.nei_agents) == 0:
            return []
        if self.all_states_aligned():
            if self.small_iter > self.max_small_iterations:
                self.small_iter = 0

            self.state = 'plan'
        send_order = self.get_regular_send_order()
        return send_order

    def process(self, observation):
        self.observe(observation)
        self.update_beliefs()

        if self.state == 'first':
            send_order = self.state_first()

        elif self.state == 'plan':
            send_order = self.state_plan()

        elif self.state == 'f_plan':
            send_order = self.state_f_plan()

        else:
            raise RuntimeError('unknown state')

        return -1, send_order


class CamsAlgAgent(AlgAgent):
    def __init__(self, sim_agent, with_breakdowns, max_small_iterations=10):
        super(CamsAlgAgent, self).__init__(sim_agent)
        self.beliefs = {}
        self.with_breakdowns = with_breakdowns

        # states
        self.sync_time = 0
        self.state_counter = 0
        self.state = 'first'

        # max-sum
        self.next_cams_pos, self.next_cams_action = None, None
        self.max_small_iterations = max_small_iterations
        self.small_iter = 0

    def reset_beliefs(self):
        self.beliefs = reset_beliefs(self.all_agents, self.nei_pos_nodes)

    def update_beliefs(self):
        update_beliefs(self.mailbox, self.step_count, self.sync_time, self.state, self.small_iter, self.beliefs)

    def get_regular_send_order(self, show_next_pos=True):
        """
        SEND ORDER: message -> [(from, to, s_time, content), ...]
        """
        messages_for_all_agents = get_messages_for_all_agents(self, show_next_pos)
        return messages_for_all_agents

    def get_messages_for_nei_pos_nodes(self):
        """
        SEND ORDER: message -> [(from, to, s_time, content), ...]
        """
        messages = []
        for pos_node in self.nei_pos_nodes:
            content = {
                'state': self.state,
                'small_iter': self.small_iter
            }
            # create ms_message
            # add from targets
            # add from other pos_nodes

            new_message = (self.name, pos_node['name'], self.step_count, content)
            messages.append(new_message)
        return messages

    def get_plan_send_order(self):
        """
        SEND ORDER: message -> [(from, to, s_time, content), ...]
        """
        messages = []
        # for all agents
        messages_for_all_agents = get_messages_for_all_agents(self, show_next_pos=False)
        messages.extend(messages_for_all_agents)

        # for pos nodes
        messages_for_nei_pos_nodes = self.get_messages_for_nei_pos_nodes()
        messages.extend(messages_for_nei_pos_nodes)

        return messages

    def decide_next_cams_move(self):
        next_action_value_dict = {}
        nei_targets = [AttributeDict(t) for t in self.nei_targets]
        for next_action, next_pos in self.pos.actions_dict.items():
            next_value = 0
            for target in nei_targets:
                if distance_nodes(next_pos, target.pos) <= self.sr:
                    if self.name in target.fmr_nei:
                        next_value += min(self.cred, target.temp_req)
            next_action_value_dict[next_action] = next_value
        max_value = max(next_action_value_dict.values())
        self.next_cams_action = random.choice([k for k, v in next_action_value_dict.items() if v == max_value])
        self.next_cams_pos = self.pos.actions_dict[self.next_cams_action]

    def update_breakdowns(self):
        if len(self.col_agents_list) > 0:
            self.next_cams_action = 404
            self.next_cams_pos = self.pos

    def all_agents_states_aligned(self):
        for agent in self.all_agents:
            agent_name = agent['name']
            state = self.beliefs[agent_name]['state']
            if state != self.state:
                return False
        return True

    def all_states_aligned(self):
        for agent in self.all_agents:
            agent_name = agent['name']
            state = self.beliefs[agent_name]['state']
            if state != self.state:
                return False
        for nei_pos_node in self.nei_pos_nodes:
            pos_node_name = nei_pos_node['name']
            state = self.beliefs[pos_node_name]['state']
            small_iter = self.beliefs[pos_node_name]['small_iter']
            if state != self.state or small_iter != self.small_iter:
                return False
        return True

    # ------------------------------------ states ------------------------------------ #

    def state_first(self):
        self.reset_beliefs()
        self.state = 'f_move'
        move_order = -1
        self.small_iter = 0
        send_order = self.get_regular_send_order(show_next_pos=False)
        return move_order, send_order

    def state_f_move(self):
        move_order = -1
        send_order = self.get_regular_send_order(show_next_pos=False)
        if self.all_agents_states_aligned():
            self.reset_beliefs()
            self.sync_time = self.step_count
            self.small_iter = 0
            self.state = 'plan'
        return move_order, send_order

    def state_plan(self):
        # TODO:
        """
        - agents and PosNodes exchange messages
        - choose the best one if there are neighbours or choose the random one otherwise
        """
        # self.decide_next_possible_move()
        self.small_iter += 1
        self.state = 'f_plan'
        move_order = -1
        self.sync_time = self.step_count
        send_order = self.get_plan_send_order()
        if self.small_iter == self.max_small_iterations:
            self.decide_next_cams_move()
        return move_order, send_order

    def state_f_plan(self):
        if self.all_states_aligned():
            if self.small_iter < self.max_small_iterations:
                self.state = 'plan'
            else:
                self.state = 'move'
                # if self.with_breakdowns:
                #     self.update_breakdowns()
                send_order = self.get_plan_send_order()
                return self.next_cams_action, send_order
        move_order = -1
        send_order = self.get_plan_send_order()
        return move_order, send_order

    def state_move(self):
        move_order = -1
        send_order = self.get_regular_send_order(show_next_pos=False)
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


class CamsAlg:
    def __init__(self, with_breakdowns=False, max_small_iterations=5):
        self.name = 'CAMS'
        self.agents, self.agents_dict = None, None
        self.pos_nodes, self.pos_nodes_dict = None, None
        self.all_entities, self.all_entities_dict = None, None
        self.sim_agents, self.sim_targets, self.sim_nodes = None, None, None
        self.with_breakdowns = with_breakdowns
        self.max_small_iterations = max_small_iterations

    def create_entities(self, sim_agents, sim_targets, sim_nodes):
        self.sim_agents, self.sim_targets, self.sim_nodes = sim_agents, sim_targets, sim_nodes
        self.agents, self.agents_dict = [], {}
        self.pos_nodes, self.pos_nodes_dict = [], {}
        self.all_entities, self.all_entities_dict = [], {}
        for sim_agent in sim_agents:
            new_agent = CamsAlgAgent(sim_agent, self.with_breakdowns, self.max_small_iterations)
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent
            self.all_entities.append(new_agent)
            self.all_entities_dict[new_agent.name] = new_agent
        for node in sim_nodes:
            new_pos_node = CamsAlgPosNode(node, self.max_small_iterations)
            self.pos_nodes.append(new_pos_node)
            self.pos_nodes_dict[new_pos_node.name] = new_pos_node
            self.all_entities.append(new_pos_node)
            self.all_entities_dict[new_pos_node.name] = new_pos_node

    def reset(self, sim_agents, sim_targets, sim_nodes):
        self.create_entities(sim_agents, sim_targets, sim_nodes)

    def get_actions(self, observations):
        print('[ALG] execute get_actions..')
        actions = {}
        # state_counter
        print(f' ------------------------------ step: {observations["step_count"]} ------------------------------ ')
        for entity in self.all_entities:
            move_order, send_order = entity.process(observations[entity.name])
            actions[entity.name] = {'move': move_order, 'send': send_order}
            # if 'agent' in entity.name:  #  and entity.state == 'plan'
            #     print(f"{entity.name}'s state counter: {entity.state_counter}, state: {entity.state}, iter: {entity.small_iter}")
        for entity in self.all_entities:
            if 'agent' in entity.name:
                print(f"{entity.name}'s state counter: {entity.state_counter}, state: {entity.state}, iter: {entity.small_iter}")
            # elif entity.is_with_nei():
            #     print(f"{entity.name}'s state counter: {entity.state_counter}, state: {entity.state}, iter: {entity.small_iter}")


            # for target in self.sim_targets:
            #     print(f'{target.name}: {target.fmr_nei=}')
        return actions

    def get_info(self):
        info = {}
        return info


def main():
    set_seed(random_seed_bool=False, i_seed=678)
    # set_seed(random_seed_bool=True)

    alg = CamsAlg()
    # alg = CamsAlg(with_breakdowns=True)

    # test_mst_alg(alg, to_render=False)
    # test_mst_alg(alg, to_render=True, plot_every=10)
    # set_seed(True, 353)
    test_mst_alg(
        alg,
        n_agents=5,
        n_targets=5,
        to_render=True,
        plot_every=10,
        n_problems=3,
        max_steps=2000,
        with_fmr=True,
    )


if __name__ == '__main__':
    main()

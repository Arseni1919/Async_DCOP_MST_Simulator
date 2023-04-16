from globals import *
from algs.test_mst_alg import test_mst_alg
from algs.alg_functions import *
from algs.alg_objects import *
from functions import *


def flatten_message(message, to_flatten=True):
    if to_flatten:
        min_value = min(message.values())
        return {pos_i: value - min_value for pos_i, value in message.items()}
    return message


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


# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# -------------------------------------------CamsAlgPosNode--------------------------------------------- #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #

class CamsAlgPosNode:
    def __init__(self, pos, max_small_iterations=10):
        self.pos = pos
        self.name = pos.xy_name
        self.step_count = None
        self.mailbox = {}
        self.beliefs = {}

        self.inf = -900000
        self.dust = {}

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

    def update_dust_weights(self):
        dust = {}
        for agent in self.all_agents:
            dust[agent['name']] = random.uniform(1e-10, 1e-5)
        return dust

    def reset_beliefs(self):
        all_entities = []
        all_entities.extend(self.all_agents)
        beliefs = {}
        for entity in all_entities:
            # create belief if this is new agent
            beliefs[entity['name']] = {
                'state': '',
                'small_iter': None,
                'weights': {}
            }
        self.beliefs = beliefs

    def update_beliefs(self):
        """
            income messages -> [(from, s_time, content), ...]
            self.mailbox[self.step_count] = observation.new_messages
            """
        new_messages = self.mailbox[self.step_count]
        for from_a_name, s_time, content in new_messages:
            # if old message -> ignore
            if s_time > self.sync_time:
                state = content['state']
                small_iter = content['small_iter']
                weights = content['weights']
                if self.state == state and self.small_iter == small_iter:
                    self.beliefs[from_a_name]['state'] = state
                    self.beliefs[from_a_name]['small_iter'] = small_iter
                    for pos_name, pos_value in weights.items():
                        self.beliefs[from_a_name]['weights'][pos_name] = pos_value

    def observe(self, observation):
        observation = AttributeDict(observation)
        self.step_count = observation.step_count
        self.pos = observation.pos
        self.nei_agents = observation.nei_agents
        self.all_agents = observation.all_agents
        # self.all_pos_nodes = observation.all_pos_nodes
        self.mailbox[self.step_count] = observation.new_messages

    def calc_belief_for_agent(self, agent):
        func_message = {pos_name: 0 for pos_name in agent['domain']}
        if len(self.nei_agents) <= 1:
            func_message[self.name] = self.dust[agent['name']]
        if len(self.nei_agents) > 2:
            func_message[self.name] = self.inf
        if len(self.nei_agents) == 2:
            domain_agent = agent['domain']
            nei_agents_copy = self.nei_agents[:]
            nei_agents_copy.remove(agent)
            other_agent = nei_agents_copy[0]
            domain_other_agent = other_agent['domain']
            for pos_name_1 in domain_agent:
                # row
                row_values = []
                for pos_name_2 in domain_other_agent:
                    # column
                    col_value = 0
                    other_agent_weights = self.beliefs[other_agent['name']]['weights']
                    if len(other_agent_weights) > 0 and pos_name_2 in other_agent_weights:
                        col_value += other_agent_weights[pos_name_2]

                    if pos_name_1 == self.name and pos_name_2 == self.name:
                        col_value = self.inf
                    elif pos_name_1 == self.name:
                        col_value += self.dust[agent['name']]
                    elif pos_name_2 == self.name:
                        col_value += self.dust[other_agent['name']]

                    row_values.append(col_value)

                func_message[pos_name_1] = max(row_values)

        return func_message

    def get_regular_send_order(self):
        messages = []
        for agent in self.nei_agents:
            content = {
                'state': self.state,
                'small_iter': self.small_iter,
                'weights': self.calc_belief_for_agent(agent)
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
        self.dust = self.update_dust_weights()
        self.state = 'plan'
        self.small_iter = 0
        return []

    def state_plan(self):
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
                self.dust = self.update_dust_weights()

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

# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ----------------------------------------------CamsAlgAgent-------------------------------------------- #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #


class CamsAlgAgent(AlgAgent):
    def __init__(self, sim_agent, with_breakdowns, max_small_iterations=10, pos_nodes=None, pos_nodes_dict=None):
        super(CamsAlgAgent, self).__init__(sim_agent)
        self.beliefs = {}
        self.with_breakdowns = with_breakdowns
        self.pos_nodes, self.pos_nodes_dict = pos_nodes, pos_nodes_dict

        # states
        self.sync_time = 0
        self.state_counter = 0
        self.state = 'first'

        # max-sum
        self.next_cams_pos, self.next_cams_action = None, None
        self.max_small_iterations = max_small_iterations
        self.small_iter = 0

    def reset_beliefs(self):
        all_entities = []
        all_entities.extend(self.all_agents)
        all_entities.extend(self.nei_pos_nodes)
        beliefs = {}
        for entity in all_entities:
            # create belief if this is new agent
            beliefs[entity['name']] = {
                'state': '',
                'small_iter': None
            }
        for nei_pos_node in self.nei_pos_nodes:
            beliefs[nei_pos_node['name']]['weights'] = {}
        self.beliefs = beliefs

    def update_beliefs(self):
        """
        income messages -> [(from, s_time, content), ...]
        self.mailbox[self.step_count] = observation.new_messages
        """
        # general
        new_messages = self.mailbox[self.step_count]
        for from_a_name, s_time, content in new_messages:
            # if old message -> ignore
            if s_time > self.sync_time:
                state = content['state']
                small_iter = content['small_iter']
                if self.state == state and self.small_iter == small_iter:
                    self.beliefs[from_a_name]['state'] = state
                    self.beliefs[from_a_name]['small_iter'] = small_iter
                    # from pos_nodes
                    if 'agent' not in from_a_name:
                        weights = content['weights']
                        for pos_name, pos_value in weights.items():
                            self.beliefs[from_a_name]['weights'][pos_name] = pos_value

    def get_regular_send_order(self, show_next_pos=True):
        """
        SEND ORDER: message -> [(from, to, s_time, content), ...]
        """
        messages_for_all_agents = get_messages_for_all_agents(self, show_next_pos)
        return messages_for_all_agents

    def create_ms_message_with_target_upload(self):
        # create new ms message
        ms_message = {nei_pos_name: 0 for nei_pos_name in self.pos.neighbours}
        ms_message[self.pos.xy_name] = 0
        # add from targets
        for nei_pos_name in self.pos.neighbours:
            nei_pos = self.pos_nodes_dict[nei_pos_name]
            for target in self.nei_targets:
                if distance_nodes(nei_pos.pos, target['pos']) <= self.sr:
                    ms_message[nei_pos_name] += min(self.cred, target['temp_req'])
        for target in self.nei_targets:
            if distance_nodes(self.pos, target['pos']) <= self.sr:
                ms_message[self.pos.xy_name] += min(self.cred, target['temp_req'])
        return ms_message

    def add_others_pos_nodes_upload(self, pos_node, ms_message):
        for other_pos_node in self.nei_pos_nodes:
            if other_pos_node['name'] != pos_node['name']:
                believed_weights = self.beliefs[other_pos_node['name']]['weights']
                if len(believed_weights) > 0:
                    for next_pos in ms_message.keys():
                        ms_message[next_pos] += believed_weights[next_pos]
        return ms_message

    def get_messages_for_nei_pos_nodes(self):
        """
        SEND ORDER: message -> [(from, to, s_time, content), ...]
        """
        messages = []
        for pos_node in self.nei_pos_nodes:
            # create ms_message + add from targets
            var_message = self.create_ms_message_with_target_upload()

            # add from other pos_nodes
            var_message = self.add_others_pos_nodes_upload(pos_node, var_message)
            var_message = flatten_message(var_message)

            content = {
                'state': self.state,
                'small_iter': self.small_iter,
                'weights': var_message,
            }

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
            # targets
            for target in nei_targets:
                if distance_nodes(next_pos, target.pos) <= self.sr:
                    if self.name in target.fmr_nei:
                        next_value += min(self.cred, target.temp_req)
            next_action_value_dict[next_action] = next_value
            # pos_nodes
            for pos_node in self.nei_pos_nodes:

                pos_value = self.beliefs[pos_node['name']]['weights'][next_pos.xy_name]
                next_value += pos_value

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


# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# -----------------------------------------------CamsAlg------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #

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
            new_agent = CamsAlgAgent(sim_agent, self.with_breakdowns, self.max_small_iterations, self.pos_nodes, self.pos_nodes_dict)
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
        # logging.debug('[ALG] execute get_actions..')
        actions = {}
        # state_counter
        print(f' ------------------------------ step: {observations["step_count"]} ------------------------------ ')
        for entity in self.all_entities:
            move_order, send_order = entity.process(observations[entity.name])
            actions[entity.name] = {'move': move_order, 'send': send_order}
            # if 'agent' in entity.name:  #  and entity.state == 'plan'
            #     print(f"{entity.name}'s state counter: {entity.state_counter}, state: {entity.state}, iter: {entity.small_iter}")
        # for entity in self.all_entities:
        #     if 'agent' in entity.name:
        #         print(f"{entity.name}'s state counter: {entity.state_counter}, state: {entity.state}, iter: {entity.small_iter}")
            # elif entity.is_with_nei():
            #     print(f"{entity.name}'s state counter: {entity.state_counter}, state: {entity.state}, iter: {entity.small_iter}")


            # for target in self.sim_targets:
            #     print(f'{target.name}: {target.fmr_nei=}')
        return actions

    def get_info(self):
        info = {}
        return info


def main():
    set_seed(random_seed_bool=False, i_seed=191)
    # set_seed(random_seed_bool=True)

    alg = CamsAlg()
    # alg = CamsAlg(with_breakdowns=True)

    # test_mst_alg(alg, to_render=False)
    # test_mst_alg(alg, to_render=True, plot_every=10)
    # set_seed(True, 353)
    test_mst_alg(
        alg,
        n_agents=20,
        n_targets=20,
        to_render=True,
        plot_every=10,
        n_problems=3,
        max_steps=2000,
        with_fmr=True,
    )


if __name__ == '__main__':
    main()

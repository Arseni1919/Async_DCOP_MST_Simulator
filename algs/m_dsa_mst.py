from globals import *
from algs.test_mst_alg import test_mst_alg
from algs.alg_functions import *
from algs.alg_objects import *


def within_sr_from_most(robot, robot_pos_name_set, target_set, pos_dict_name_pos):
    within_sr_range_dict = {}
    max_list = []
    for robot_name in robot_pos_name_set:
        count = sum([distance(target.pos.xy_pos, pos_dict_name_pos[robot_name]) < robot.sr for target in target_set])
        max_list.append(count)
        within_sr_range_dict[robot_name] = count
    max_value = max(max_list)

    within_sr_range_list, target_set_to_send = [], []
    for robot_name, count in within_sr_range_dict.items():
        if count == max_value:
            within_sr_range_list.append(robot_name)
            target_set_to_send.extend(list(filter(
                lambda x: distance(x.pos.xy_pos, pos_dict_name_pos[robot_name]) < robot.sr,
                target_set
            )))
    target_set_to_send = list(set(target_set_to_send))
    return within_sr_range_list, target_set_to_send


def select_pos_internal(robot, robot_pos_name_set, funcs, pos_dict_name_pos):
    max_func_value = max([target.req for target in funcs]) if len(funcs) > 0 else 0
    if len(robot_pos_name_set) == 1 or max_func_value < 1:
        return random.sample(robot_pos_name_set, 1)[0]
    target_set = []
    for target in funcs:
        if target.req == max_func_value:
            if any([distance(target.pos.xy_pos, pos_dict_name_pos[p_n]) < robot.sr for p_n in robot_pos_name_set]):
                target_set.append(target)

    if len(target_set) == 0:
        return random.sample(robot_pos_name_set, 1)[0]

    within_sr_range_list, target_set = within_sr_from_most(robot, robot_pos_name_set, target_set, pos_dict_name_pos)
    for target in target_set:
        funcs.remove(target)

    return select_pos_internal(robot, within_sr_range_list, funcs, pos_dict_name_pos)


class DsaMstAlgAgent(AlgAgent):
    def __init__(self, sim_agent):
        super(DsaMstAlgAgent, self).__init__(sim_agent)

    def get_move_order(self):
        # new_pos =select_pos_internal()
        return random.randint(0, 4)

    def get_send_order(self):
        return []

    def process(self, observation):
        # TODO
        self.observe(observation)
        move_order = self.get_move_order()
        send_order = self.get_send_order()
        return move_order, send_order


class DsaMstAlg:
    def __init__(self):
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
    test_mst_alg(alg, to_render=False)


if __name__ == '__main__':
    main()

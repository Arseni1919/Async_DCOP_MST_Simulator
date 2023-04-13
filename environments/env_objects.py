from globals import *


def distance_nodes(node1, node2):
    return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)


class Node:
    def __init__(self, x, y, t=0, neighbours=None, new_ID=None):
        if new_ID:
            self.ID = new_ID
        else:
            self.ID = f'{x}_{y}_{t}'
        self.xy_name = f'{x}_{y}'
        self.x = x
        self.y = y
        self.xy_pos = (self.x, self.y)
        self.t = t
        if neighbours is None:
            self.neighbours = []
        else:
            self.neighbours = neighbours
        # self.neighbours = neighbours
        self.actions_dict = {}

        self.h = 0
        self.g = t
        self.parent = None
        self.g_dict = {}

    def f(self):
        return self.t + self.h
        # return self.g + self.h

    def reset(self, target_nodes=None, **kwargs):
        if 'start_time' in kwargs:
            self.t = kwargs['start_time']
        else:
            self.t = 0
        self.h = 0
        self.g = self.t
        self.ID = f'{self.x}_{self.y}_{self.t}'
        self.parent = None
        if target_nodes is not None:
            self.g_dict = {target_node.xy_name: 0 for target_node in target_nodes}
        else:
            self.g_dict = {}

    def init_actions(self, nodes_dict):
        # idle
        self.actions_dict[0] = self

        # new pos
        x = self.x
        y = self.y
        new_pos_name = f'{x}_{y + 1}'
        if new_pos_name in nodes_dict:
            self.actions_dict[1] = nodes_dict[new_pos_name]

        new_pos_name = f'{x + 1}_{y}'
        if new_pos_name in nodes_dict:
            self.actions_dict[2] = nodes_dict[new_pos_name]

        new_pos_name = f'{x}_{y - 1}'
        if new_pos_name in nodes_dict:
            self.actions_dict[3] = nodes_dict[new_pos_name]

        new_pos_name = f'{x - 1}_{y}'
        if new_pos_name in nodes_dict:
            self.actions_dict[4] = nodes_dict[new_pos_name]


class SimTarget:
    def __init__(self, num, pos, req, life_start, life_end):
        self.num = num
        self.pos = pos
        self.req = req
        self.temp_req = req
        self.life_start = life_start
        self.life_end = life_end
        self.fmr_nei = []

        self.name = f'target_{self.num}'

    def update_temp_req(self, agents):
        temp_req = self.req
        for agent in agents:
            if distance_nodes(self.pos, agent.pos) <= agent.sr:
                temp_req = max(0, temp_req - agent.cred)
        self.temp_req = temp_req

    def reset(self):
        pass


class SimAgent:
    def __init__(self, num, cred=20, sr=10, mr=1, pos=None):
        self.num = num
        self.cred = cred
        self.sr = sr
        self.mr = mr

        self.pos = pos
        self.start_pos = self.pos
        self.prev_pos = None
        self.last_time_pos = None

        self.next_pos = None
        self.is_moving = False
        self.arrival_time = None

        self.col_agents_list = None
        self.is_broken = False
        self.broken_pos = None
        self.broken_time = -1

        self.name = f'agent_{self.num}'

    def reset(self):
        self.pos = self.start_pos
        self.prev_pos = self.start_pos
        self.next_pos = None
        self.arrival_time = -1
        self.is_broken = False
        self.broken_pos = None
        self.broken_time = -1

    def clear_col_agents_list(self):
        self.col_agents_list = []

    def set_next_pos_and_time(self, next_pos, arrival_time):
        self.next_pos = next_pos
        self.arrival_time = arrival_time
        self.is_moving = True

    def go_to_next_pos(self):
        if self.next_pos is None:
            raise RuntimeError('next_pos is None')
        self.prev_pos = self.pos
        self.pos = self.next_pos
        self.next_pos = None
        self.arrival_time = -1
        self.is_moving = False

    def get_broken(self, pos, t):
        if not self.is_broken:
            self.is_broken = True
            self.broken_pos = pos
            self.broken_time = t
        else:
            raise RuntimeError(f'{self.name} is already broken in pos: {self.broken_pos} at time {self.broken_time}')

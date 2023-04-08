from globals import *


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


class SimTarget:
    def __init__(self, num, pos, req, life_start, life_end):
        self.num = num
        self.pos = pos
        self.req = req
        self.life_start = life_start
        self.life_end = life_end

        self.name = f'target_{self.num}'


class SimAgent:
    def __init__(self, num, cred=20, sr=10, mr=1, pos=None):
        self.num = num
        self.cred = cred
        self.sr = sr
        self.mr = mr
        self.pos = pos
        self.start_pos = self.pos
        self.prev_pos = pos
        self.is_broken = False
        self.broken_pos = None
        self.broken_time = -1

        self.name = f'agent_{self.num}'

    def reset(self):
        self.pos = self.start_pos
        self.prev_pos = self.start_pos
        self.is_broken = False
        self.broken_pos = None
        self.broken_time = -1

    def get_broken(self, pos, t):
        if not self.is_broken:
            self.is_broken = True
            self.broken_pos = pos
            self.broken_time = t
        else:
            raise RuntimeError(f'{self.name} is already broken in pos: {self.broken_pos} at time {self.broken_time}')

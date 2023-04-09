from globals import *


class AlgAgent:
    def __init__(self, sim_agent):
        self.num = sim_agent.num
        self.name = sim_agent.name
        self.cred = sim_agent.cred
        self.sr = sim_agent.sr
        self.mr = sim_agent.mr

        self.pos = sim_agent.pos
        self.start_pos = sim_agent.start_pos
        self.prev_pos = None
        self.next_pos = None
        self.is_moving = False
        self.is_broken = False
        self.broken_pos = None
        self.broken_time = -1

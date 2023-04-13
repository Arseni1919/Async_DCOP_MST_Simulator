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
        self.col_agents_list = None

        self.step_count = None
        self.nei_targets = None
        self.nei_agents = None
        self.all_agents = None
        self.mailbox = {}

    def observe(self, observation):
        observation = AttributeDict(observation)
        self.step_count = observation.step_count
        self.cred = observation.cred
        self.sr = observation.sr
        self.mr = observation.mr
        self.pos = observation.pos
        self.start_pos = observation.start_pos
        self.prev_pos = observation.prev_pos
        self.next_pos = observation.next_pos
        self.is_moving = observation.is_moving
        self.is_broken = observation.is_broken
        self.broken_pos = observation.broken_pos
        self.broken_time = observation.broken_time
        self.col_agents_list = observation.col_agents_list
        self.nei_targets = observation.nei_targets
        self.nei_agents = observation.nei_agents
        self.all_agents = observation.all_agents
        self.mailbox[self.step_count] = observation.new_messages

import logging

import matplotlib.pyplot as plt

from globals import *
from environments.env_functions import *
from plot_functions.plot_functions import *


class AsyncDcopMstEnv:
    def __init__(self, max_steps, map_dir, with_fmr=False, to_render=True):
        self.max_steps = max_steps
        self.map_dir = map_dir
        self.with_fmr = with_fmr
        self.to_render = to_render
        self.name = 'AsyncDcopMstEnv'
        # create_new_problem
        self.map_np, self.height, self.width, self.nodes, self.nodes_dict = None, None, None, None, None
        self.agents, self.agents_dict = None, None
        self.targets, self.targets_dict = None, None
        # reset
        self.step_count = None
        self.mailbox = None

        # for rendering
        self.amount_of_messages_list = None
        # self.fig, self.ax = plt.subplots(2, 2, figsize=(12, 8))
        if self.to_render:
            self.fig, self.ax = plt.subplot_mosaic("AAB;AAC;AAD", figsize=(12, 8))

    def create_new_problem(self, path='maps', n_agents=2, n_targets=2):
        self.map_np, (self.height, self.width) = get_np_from_dot_map(self.map_dir, path=path)
        self.nodes, self.nodes_dict = build_graph_from_np(self.map_np, show_map=False)
        self.agents, self.agents_dict = [], {}
        self.targets, self.targets_dict = [], {}
        positions_pool = random.sample(self.nodes, n_agents + n_targets)

        # create agents
        for i in range(n_agents):
            new_pos = positions_pool.pop()
            new_agent = SimAgent(num=i, cred=30, sr=10, mr=1, pos=new_pos)
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent
            # print(f'{new_agent.name} - {new_agent.pos.x}-{new_agent.pos.y}')
            # print(len(positions_pool))

        # create targets
        for i in range(n_targets):
            new_pos = positions_pool.pop()
            new_target = SimTarget(num=i, pos=new_pos, req=100, life_start=0, life_end=self.max_steps)
            self.targets.append(new_target)
            self.targets_dict[new_target.name] = new_target
            # print(f'{new_target.name} - {new_target.pos.x}-{new_target.pos.y}')
            # print(len(positions_pool))

        print()

    def reset(self):
        # reset agents
        _ = [agent.reset() for agent in self.agents]

        # reset targets
        _ = [target.reset() for target in self.targets]
        _ = [target.update_temp_req(self.agents) for target in self.targets]

        # step count
        self.step_count = 0

        # mailbox
        self.mailbox = {
            agent.name: {i: [] for i in range(self.max_steps)}
            for agent in self.agents
        }
        self.mailbox.update({
            node.xy_name: {i: [] for i in range(self.max_steps)}
            for node in self.nodes
        })

        # for rendering
        self.amount_of_messages_list = []

    def update_collisions(self):
        _ = [agent.clear_col_agents_list() for agent in self.agents]
        for agent_1, agent_2 in combinations(self.agents, 2):
            if agent_1.pos.xy_name == agent_2.pos.xy_name:
                agent_1.col_agents_list.append(agent_2.name)
                agent_2.col_agents_list.append(agent_1.name)

    def update_fmr_nei(self):
        if self.with_fmr:
            for agent in self.agents:
                if agent.last_time_pos == agent.pos:
                    return
            for target in self.targets:
                target.fmr_nei = select_FMR_nei(target, self.targets, self.agents, self.nodes_dict)
            # print()

    def get_nei_targets(self, agent):
        logging.debug(f"[ENV]: get_nei_targets")
        def is_close(t):
            for next_pos_name in agent.pos.neighbours:
                next_pos = self.nodes_dict[next_pos_name]
                if distance_nodes(next_pos, t.pos) <= agent.sr:
                    return True
            return False

        nei_targets = []
        for target in self.targets:
            if is_close(target):
                nei_targets.append({
                    'name': target.name,
                    'num': target.num,
                    'pos': target.pos,
                    'req': target.req,
                    'temp_req': target.temp_req,
                    'fmr_nei': [a.name for a in target.fmr_nei]
                })
        return nei_targets

    def get_pos_node_nei_agents(self, pos_node):
        logging.debug(f"[ENV]: get_pos_node_nei_agents")
        pos_name = pos_node.xy_name
        nei_agents = []
        for other_agent in self.agents:
            if pos_name == other_agent.pos.xy_name or pos_name in other_agent.pos.neighbours:
                nei_agents.append({
                    'name': other_agent.name,
                    'num': other_agent.num,
                    'pos': other_agent.pos,
                })
        return nei_agents

    def get_nei_agents(self, agent):
        logging.debug(f"[ENV]: get_nei_agents")
        nei_agents = []
        for other_agent in self.agents:
            if agent.name != other_agent.name:
                if distance_nodes(agent.pos, other_agent.pos) <= agent.sr + agent.mr + other_agent.sr + other_agent.mr:
                    nei_agents.append({
                        'name': other_agent.name,
                        'num': other_agent.num,
                        'pos': other_agent.pos,
                    })
        return nei_agents

    def get_nei_pos_nodes(self, agent):
        logging.debug(f"[ENV]: get_nei_pos_nodes")
        nei_pos_nodes = []
        for pos_node_name in agent.pos.neighbours:
            nei_pos_nodes.append({
                'name': pos_node_name,
                'pos': self.nodes_dict[pos_node_name],
            })
        nei_pos_nodes.append({
            'name': agent.pos.xy_name,
            'pos': agent.pos,
        })
        return nei_pos_nodes

    def get_all_agents(self, agent=None):
        logging.debug(f"[ENV]: get_all_agents")
        all_agents = []
        for other_agent in self.agents:
            if agent is not None:
                if agent.name == other_agent.name:
                    continue
            all_agents.append({
                    'name': other_agent.name,
                    'num': other_agent.num,
                    'pos': other_agent.pos,
                })
        return all_agents

    def get_all_pos_nodes(self, pos_node=None):
        # logging.debug(f"[ENV]: get_all_pos_nodes")
        # print(f"[ENV]: get_all_pos_nodes")
        all_pos_nodes = []
        # for i_pos_node in self.nodes:
        #     if pos_node is not None:
        #         if pos_node.xy_name == i_pos_node.xy_name:
        #             continue
        #     all_pos_nodes.append({
        #         'name': i_pos_node.xy_name,
        #     })
        # logging.debug(f"[ENV]: finished get_all_pos_nodes")
        return all_pos_nodes

    def get_observations(self):
        print("[ENV] execute get_observations..")
        # observations for agents
        observations = {
            agent.name: {
                'name': agent.name,
                'step_count': self.step_count,
                'cred': agent.cred,
                'sr': agent.sr,
                'mr': agent.mr,
                'pos': agent.pos,
                'start_pos': agent.start_pos,
                'prev_pos': agent.prev_pos,
                'next_pos': agent.next_pos,
                'is_moving': agent.is_moving,
                'is_broken': agent.is_broken,
                'broken_pos': agent.broken_pos,
                'broken_time': agent.broken_time,
                'col_agents_list': agent.col_agents_list,
                'nei_targets': self.get_nei_targets(agent),
                'nei_agents': self.get_nei_agents(agent),
                'all_agents': self.get_all_agents(agent),
                'nei_pos_nodes': self.get_nei_pos_nodes(agent),
                # 'all_pos_nodes': self.get_all_pos_nodes(),
                'new_messages': self.mailbox[agent.name][self.step_count],
            }
            for agent in self.agents
        }
        # observations for pos_nodes
        for node in self.nodes:
            observations[node.xy_name] = {
                'name': node.xy_name,
                'step_count': self.step_count,
                'pos': node,
                'nei_agents': self.get_pos_node_nei_agents(node),
                'all_agents': self.get_all_agents(),
                # 'all_pos_nodes': self.get_all_pos_nodes(node),
                'new_messages': self.mailbox[node.xy_name][self.step_count],
            }
        # general info
        observations['step_count'] = self.step_count
        logging.debug("[ENV] finished get_observations.")
        return observations

    def execute_move_order(self, agent, move_order):
        """
        MOVE ORDER: -1 - wait, 0 - stay, 1 - up, 2 - right, 3 - down, 4 - left
        """
        # --- broken ---
        if agent.is_broken:
            return False

        # --- arrived ---
        if self.step_count == agent.arrival_time:
            agent.go_to_next_pos()

        # --- still moving or waiting ---
        if agent.is_moving:
            return False

        # get broken
        if move_order == 404:
            agent.get_broken(agent.pos, self.step_count)

        # --- if not moving and not broken... ---
        if move_order is None:
            raise RuntimeError('move_order is None')

        # idle
        if move_order == 0:
            agent.last_time_pos = agent.pos
            return True

        # new pos
        if move_order in agent.pos.actions_dict:
            new_pos = agent.pos.actions_dict[move_order]
            time_to_arrive = self.step_count + random.randint(2, 3)  # takes time to make a movement
            agent.set_next_pos_and_time(next_pos=new_pos, arrival_time=time_to_arrive)
        return True

    def execute_send_order(self, send_order):
        """
        SEND ORDER: message -> [(from, to, s_time, content), ...]
        """
        for from_a_name, to_a_name, s_time, content in send_order:
            # time_to_arrive = min(self.max_steps - 1, self.step_count + 1 + random.randint(0, 3))
            time_to_arrive = self.step_count + random.randint(1, 1)
            if time_to_arrive < self.max_steps:
                self.mailbox[to_a_name][time_to_arrive].append((from_a_name, s_time, content))

    def step(self, actions):
        """
        ACTION:
            MOVE ORDER: -1 - wait, 0 - stay, 1 - up, 2 - right, 3 - down, 4 - left
            SEND ORDER: message -> [(from, to, s_time, content), ...]
        """
        print('[ENV] execute step..')
        # move agents + send agents' messages
        print("[ENV] move agents + send agents' messages..")
        for agent in self.agents:
            move_order = actions[agent.name]['move']
            send_order = actions[agent.name]['send']
            self.execute_move_order(agent, move_order)
            self.execute_send_order(send_order)

        # send pos_nodes' messages
        print("[ENV] send pos_nodes' messages..")
        for node in self.nodes:
            if node.xy_name in actions:
                send_order = actions[node.xy_name]['send']
                self.execute_send_order(send_order)

        # update targets' data
        _ = [target.update_temp_req(self.agents) for target in self.targets]
        self.update_fmr_nei()

        # update agents' data
        self.update_collisions()

        # for rendering
        self.amount_of_messages_list.append(self.calc_amount_of_messages())

        self.step_count += 1

    def calc_amount_of_messages(self):
        amount_of_messages = 0
        for agent in self.agents:
            for i in range(self.step_count):
                amount_of_messages += len(self.mailbox[agent.name][i])
        return amount_of_messages

    def render(self, info):
        if self.to_render:
            info = AttributeDict(info)
            if info.i_time % info.plot_every == 0 or info.i_time == self.max_steps - 1:
                info.update({
                    'width': self.width,
                    'height': self.height,
                    'nodes': self.nodes,
                    'targets': self.targets,
                    'agents': self.agents,
                    'aom': self.amount_of_messages_list,
                })

                plot_async_mst_field(self.ax['A'], info)

                if 'col' in info:
                    plot_collisions(self.ax['B'], info)

                if 'cov' in info:
                    plot_rem_cov_req(self.ax['C'], info)

                plot_aom(self.ax['D'], info)

                plt.pause(0.001)
                # plt.show()

    def close(self):
        pass

    def sample_actions(self, observations):
        actions = {agent.name: {
            'move': random.randint(0, 4),
            'send': []
        }
            for agent in self.agents
        }
        return actions


def main():
    max_steps = 120
    n_problems = 3
    plot_every = 10

    info = {'plot_every': plot_every}

    # map_dir = 'empty-48-48.map'  # 48-48
    # map_dir = 'random-64-64-10.map'  # 64-64
    map_dir = 'warehouse-10-20-10-2-1.map'  # 63-161
    # map_dir = 'lt_gallowstemplar_n.map'  # 180-251

    n_agents = 10
    n_targets = 10

    env = AsyncDcopMstEnv(
        max_steps=max_steps,
        map_dir=map_dir,
    )

    for i_problem in range(n_problems):
        env.create_new_problem(path='../maps', n_agents=n_agents, n_targets=n_targets)

        # loop on algs

        env.reset()

        for i_time in range(env.max_steps):

            # env - get observations
            observations = env.get_observations()

            # alg - calc actions
            actions = env.sample_actions(observations)

            # env - make a step
            env.step(actions)

            # stats
            pass

            # render
            info['i_problem'] = i_problem
            info['i_time'] = i_time
            env.render(info)


if __name__ == '__main__':
    main()

import matplotlib.pyplot as plt

from globals import *
from environments.env_functions import *
from plot_functions.plot_functions import *


class AsyncDcopMstEnv:
    def __init__(self, max_steps, map_dir, to_render=True):
        self.max_steps = max_steps
        self.map_dir = map_dir
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
            new_agent = SimAgent(num=i, cred=20, sr=10, mr=1, pos=new_pos)
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

    def get_nei_targets(self, agent):
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
                })
        return nei_targets

    def get_nei_agents(self, agent):
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

    def get_all_agents(self, agent):
        all_agents = []
        for other_agent in self.agents:
            if agent.name != other_agent.name:
                all_agents.append({
                        'name': other_agent.name,
                        'num': other_agent.num,
                        'pos': other_agent.pos,
                    })
        return all_agents

    def get_observations(self):
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
                'nei_targets': self.get_nei_targets(agent),
                'nei_agents': self.get_nei_agents(agent),
                'all_agents': self.get_all_agents(agent),
                'new_messages': self.mailbox[agent.name][self.step_count],
            }
            for agent in self.agents
        }
        observations['step_count'] = self.step_count
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

        # --- if not moving and not broken... ---
        # idle
        if move_order == 0:
            return True

        # new pos
        if move_order in agent.pos.actions_dict:
            new_pos = agent.pos.actions_dict[move_order]
            time_to_arrive = self.step_count + random.randint(5, 25)  # takes time to make a movement
            agent.set_next_pos_and_time(next_pos=new_pos, arrival_time=time_to_arrive)
        return True

    def execute_send_order(self, send_order):
        """
        SEND ORDER: message -> [(from, to, s_time, content), ...]
        """
        for from_a_name, to_a_name, s_time, content in send_order:
            # time_to_arrive = min(self.max_steps - 1, self.step_count + 1 + random.randint(0, 3))
            time_to_arrive = self.step_count + random.randint(1, 10)
            if time_to_arrive < self.max_steps:
                self.mailbox[to_a_name][time_to_arrive].append((from_a_name, s_time, content))

    def step(self, actions):
        """
        ACTION:
            MOVE ORDER: -1 - wait, 0 - stay, 1 - up, 2 - right, 3 - down, 4 - left
            SEND ORDER: message -> [(from, to, s_time, content), ...]
        """
        # move agents + send messages
        for agent in self.agents:
            move_order = actions[agent.name]['move']
            send_order = actions[agent.name]['send']
            self.execute_move_order(agent, move_order)
            self.execute_send_order(send_order)

        # update targets' data
        _ = [target.update_temp_req(self.agents) for target in self.targets]

        self.step_count += 1

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
                })

                plot_async_mst_field(self.ax['A'], info)

                if 'col' in info:
                    plot_collisions(self.ax['B'], info)

                if 'cov' in info:
                    plot_rem_cov_req(self.ax['C'], info)

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

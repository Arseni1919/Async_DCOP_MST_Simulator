import matplotlib.pyplot as plt

from globals import *
from environments.env_functions import *
from plot_functions.plot_functions import *


class AsyncDcopMstEnv:
    def __init__(self, max_steps, map_dir):
        self.max_steps = max_steps
        self.map_dir = map_dir
        self.name = 'AsyncDcopMstEnv'
        self.map_np, self.height, self.width, self.nodes, self.nodes_dict = None, None, None, None, None
        self.agents, self.agents_dict = [], {}
        self.targets, self.targets_dict = [], {}
        self.step_count = 0
        self.mailbox = {}

        # for rendering
        # self.fig, self.ax = plt.subplots(2, 2, figsize=(12, 8))
        self.fig, self.ax = plt.subplot_mosaic("AAB;AAC;AAD", figsize=(12, 8))

    def create_new_problem(self, path='maps', n_agents=2, n_targets=2):
        self.map_np, (self.height, self.width) = get_np_from_dot_map(self.map_dir, path=path)
        self.nodes, self.nodes_dict = build_graph_from_np(self.map_np, show_map=False)
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

        # step count
        self.step_count = 0

        # mailbox
        self.mailbox = {
            agent.name: {i: [] for i in range(self.max_steps)}
            for agent in self.agents
        }

    def get_observations(self):
        observations = {
            agent.name: {
                'cred': agent.cred,
                'sr': agent.sr,
                'mr': agent.mr,
                'pos': agent.pos,
                'start_pos': agent.start_pos,
                'prev_pos': agent.prev_pos,
                'is_broken': agent.is_broken,
                'broken_pos': agent.broken_pos,
                'broken_time': agent.broken_time,
                'step_count': self.step_count,
                'nei_targets': [],  # TODO
                'nei_agents': [],  # TODO
                'new_messages': None,  # TODO
            }
            for agent in self.agents
        }
        return observations

    def execute_move_order(self, agent, move_order):
        # TODO: to insert move duration
        """
        MOVE ORDER: 0 - stay, 1 - up, 2 - right, 3 - down, 4 - left
        """
        x = agent.pos.x
        y = agent.pos.y
        if move_order == 1:
            y += 1
        elif move_order == 2:
            x += 1
        elif move_order == 3:
            y -= 1
        elif move_order == 4:
            x -= 1
        elif move_order == 0:
            pass
        else:
            raise RuntimeError('wrong move order')

        new_pos_name = f'{x}_{y}'
        if new_pos_name in self.nodes_dict:
            new_pos = self.nodes_dict[new_pos_name]
            agent.goto(new_pos)

    def execute_send_order(self, send_order):
        """
        SEND ORDER: message -> [(from, to, content), ...]
        """
        for from_a_name, to_a_name, content in send_order:
            time_to_arrive = min(self.max_steps - 1, self.step_count + 1 + random.randint(0, 3))
            self.mailbox[to_a_name][time_to_arrive].append((from_a_name, content))

    def step(self, actions):
        """
        ACTION:
            MOVE ORDER: 0 - stay, 1 - up, 2 - right, 3 - down, 4 - left
            SEND ORDER: message -> [(from, to, content), ...]
        """
        for agent in self.agents:
            move_order = actions[agent.name]['move']
            send_order = actions[agent.name]['send']
            self.execute_move_order(agent, move_order)
            self.execute_send_order(send_order)
        self.step_count += 1

    def render(self, info):
        info.update({
            'width': self.width,
            'height': self.height,
            'nodes': self.nodes,
            'targets': self.targets,
            'agents': self.agents,
        })

        plot_async_mst_field(self.ax['A'], info)

        plt.pause(0.001)
        # plt.show()

    def close(self):
        pass

    def sample_actions(self, observations):
        """
        ACTION:
            MOVE ORDER: 0 - stay, 1 - up, 2 - right, 3 - down, 4 - left
            SEND ORDER: message -> [(from, to, content), ...]
        """
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

    info = {}

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

import numpy as np

from globals import *


def plot_rewards(ax, info):
    ax.cla()
    real_rewards = info['real_rewards']
    obs_rewards = info['obs_rewards']
    # real_rewards_before_action = info['real_rewards_before_action']

    max_steps = info['max_steps']
    ax.plot(real_rewards, label='real_rewards')
    ax.plot(obs_rewards, label='obs_rewards')
    # ax.plot(real_rewards_before_action, label='real_rewards_before_action')
    ax.set_xlim(0, max_steps)
    ax.set_title('Rewards')
    ax.legend()


def plot_algs_rewards(ax, info):
    ax.cla()
    max_steps = info['max_steps']
    for alg in info['algs']:
        ax.plot(info['rewards'][alg], label=f'{alg}')

    ax.set_xlim(0, max_steps)
    ax.set_title('Rewards')
    ax.legend()


def plot_async_mst_field(ax, info):
    ax.cla()
    info = AttributeDict(info)
    # field_np = np.zeros((self.height, self.width))
    field_np = np.zeros((info.width, info.height))
    # add map
    nodes_x, nodes_y = [], []
    for node in info.nodes:
        # field_np[node.x, node.y] = 1
        field_np[node.y, node.x] = 1
        nodes_x.append(node.x)
        nodes_y.append(node.y)

    ax.imshow(field_np, origin='lower', cmap='gray')
    # self.ax['A'].scatter(nodes_x, nodes_y, marker='s', s=14, color='k')
    # self.ax['A'].scatter(nodes_y, nodes_x, marker='s', s=14, color='k')

    # add targets
    targets_x, targets_y = [], []
    for target in info.targets:
        targets_x.append(target.pos.x)
        targets_y.append(target.pos.y)
    ax.scatter(targets_x, targets_y, marker='s', s=14, color='red')
    # self.ax['A'].scatter(targets_y, targets_x, marker='s', s=14, color='red')

    # add agents
    agents_x, agents_y, agents_area = [], [], []
    for agent in info.agents:
        agents_x.append(agent.pos.x)
        agents_y.append(agent.pos.y)
        agents_area.append(agent.sr)
        circle1 = plt.Circle((agent.pos.x, agent.pos.y), agent.sr, color='blue', alpha=0.1)
        # circle1 = plt.Circle((agent.pos.y, agent.pos.x), agent.sr, color='blue', alpha=0.2)
        ax.add_patch(circle1)
    ax.scatter(agents_x, agents_y, marker='o', s=14, color='blue')
    # self.ax['A'].scatter(agents_y, agents_x, marker='o', s=14, color='blue')

    # self.ax['A'].scatter(agents_y, agents_x, marker='o', s=agents_area, color='blue', alpha=0.2)

    ax.set_ylim(0, info.width)
    ax.set_xlim(0, info.height)
    ax.set_title(f'prob: {info.i_problem + 1}, iter: {info.i_time}')


def plot_collisions(ax, info):
    ax.cla()
    info = AttributeDict(info)

    ax.plot(np.cumsum(info.col))
    ax.set_xlim(0, info.max_steps)
    ax.set_title('collisions')


def plot_rem_cov_req(ax, info):
    ax.cla()
    info = AttributeDict(info)

    ax.plot(info.cov)
    ax.set_xlim(0, info.max_steps)
    ax.set_title('Remained Coverage Req.')


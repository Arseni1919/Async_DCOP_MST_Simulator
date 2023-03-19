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

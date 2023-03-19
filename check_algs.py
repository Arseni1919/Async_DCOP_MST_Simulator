from globals import *
from functions import *
from environments.async_dcop import AsyncDCOP
from algs.dsa import AlgDSA
from algs.dsa_async import AlgAsyncDSA
from plot_functions.plot_functions import *


def main():
    # random_seed_bool = False
    random_seed_bool = True
    set_seed(random_seed_bool)

    n_agents = 10
    domain_size = 5
    constraints_type = 'clique'

    n_problems = 1

    env = AsyncDCOP(
        max_steps=120,
        n_agents=n_agents, domain_size=domain_size, constraints_type=constraints_type
    )
    algs = [AlgDSA(), AlgAsyncDSA()]

    # for render
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    plot_info = {
        'rewards': {alg.name: [] for alg in algs},
        'algs': [alg.name for alg in algs],
        'max_steps': env.max_steps,
    }

    for i_problem in range(n_problems):
        env.create_new_problem()
        for alg in algs:
            observation, info = env.reset()
            alg.reset(env.agents)

            for i in range(env.max_steps):

                # choose actions
                actions = alg.calc_actions(observation)

                # execute actions
                new_observation, rewards, terminated, truncated, info = env.step(actions)

                # after actions
                if terminated or truncated:
                    observation, info = env.reset()
                else:
                    observation = new_observation

                # stats
                pass

            # for render
            plot_info['rewards'][alg.name] = env.h_real_rewards

            plot_algs_rewards(ax[0, 0], plot_info)
            plt.pause(0.001)

    plt.show()
    env.close()


if __name__ == '__main__':
    main()

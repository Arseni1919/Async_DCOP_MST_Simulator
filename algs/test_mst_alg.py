from environments.async_dcop_mst import AsyncDcopMstEnv


def test_mst_alg(alg, to_render=True):
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
        to_render=to_render,
    )

    for i_problem in range(n_problems):
        env.create_new_problem(path='../maps', n_agents=n_agents, n_targets=n_targets)
        alg.create_entities(env.agents, env.targets)

        # loop on algs

        env.reset()
        alg.reset(env.agents, env.targets)

        for i_time in range(env.max_steps):
            # env - get observations
            observations = env.get_observations()

            # alg - calc actions
            actions = alg.get_actions(observations)

            # env - make a step
            env.step(actions)

            # stats
            pass

            # render
            # global
            info['i_problem'] = i_problem
            info['i_time'] = i_time
            # from alg
            alg_info = alg.get_info()
            info.update(alg_info)
            if to_render:
                env.render(info)


def main():
    pass


if __name__ == '__main__':
    main()

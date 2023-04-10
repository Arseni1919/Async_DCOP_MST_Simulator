from globals import *


def set_seed(random_seed_bool, i_seed=1):
    if random_seed_bool:
        seed = random.randint(0, 1000)
    else:
        seed = i_seed
    random.seed(seed)
    np.random.seed(seed)
    print(f'[SEED]: --- {seed} ---')
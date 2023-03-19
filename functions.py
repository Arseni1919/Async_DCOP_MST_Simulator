from globals import *


def set_seed(random_seed_bool):
    if random_seed_bool:
        seed = random.randint(0, 1000)
    else:
        seed = 212
    random.seed(seed)
    np.random.seed(seed)
    print(f'[SEED]: --- {seed} ---')
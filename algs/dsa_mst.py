from globals import *
from algs.test_mst_alg import test_mst_alg


class DsaMstAlgAgent:
    def __init__(self):
        pass


class DsaMstAlg:
    def __init__(self):
        pass

    def create_entities(self, agents, targets):
        pass

    def reset(self, agents, targets):
        pass

    def get_actions(self, observations):
        actions = {}
        return actions

    def get_info(self):
        info = {}
        return info


def main():
    alg = DsaMstAlg()
    test_mst_alg(alg)


if __name__ == '__main__':
    main()

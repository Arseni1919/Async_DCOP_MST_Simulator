from globals import *


# class SomeEnv:
#     def __init__(self, max_steps=120):
#         self.max_steps = max_steps
#
#     def reset(self, params=None):
#         if params is None:
#             params = {}
#         observation, info = None, None
#         return observation, info
#
#     def sample_actions(self):
#         actions = {}
#         return actions
#
#     def step(self, actions):
#         observation, reward, terminated, truncated, info = {}, {}, False, False, {}
#         return observation, reward, terminated, truncated, info
#
#     def close(self):
#         pass
#
#     def render(self):
#         pass
#
#
# def main():
#     env = SomeEnv(max_steps=120)
#     observation, info = env.reset()
#     for _ in range(env.max_steps):
#
#         # choose actions
#         action = env.sample_actions()
#
#         # execute actions
#         new_observation, reward, terminated, truncated, info = env.step(action)
#
#         # after actions
#         if terminated or truncated:
#             observation, info = env.reset()
#         else:
#             observation = new_observation
#
#         # render
#         env.render()
#
#         # stats
#         pass
#
#     env.close()
#
#
# if __name__ == '__main__':
#     main()

# import gymnasium as gym
# env = gym.make("LunarLander-v2", render_mode="human")
# observation, info = env.reset(seed=42)
# for _ in range(1000):
#    action = env.action_space.sample()  # this is where you would insert your policy
#    observation, reward, terminated, truncated, info = env.step(action)
#
#    if terminated or truncated:
#       observation, info = env.reset()
# env.close()

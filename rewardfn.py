import numpy as np
from collections import defaultdict
from catanatron.state_functions import get_actual_victory_points

class InitialRewardFunction:
    def __init__(self, vp_weight = 0.1, win = 1, loss = -1):
        self.vp_weight = vp_weight
        self.loss_reward = loss
        self.win_reward = win

    def reset(self, core, game):
        self.prev_vp = defaultdict(float)

    def __call__(self, core, game):
        rewards = np.zeros(core.n_agents, dtype=np.float32)

        for i, color in enumerate(core.colors):
            current = get_actual_victory_points(game.state, color)
            rewards[i] = self.vp_weight * (current - self.prev_vp[color])
            self.prev_vp[color] = current

        winner = game.winning_color()
        if winner is not None:
            rewards[:] += self.loss_reward
            winning_index = core.colors.index(winner)
            rewards[winning_index] += self.win_reward - self.loss_reward

        return rewards
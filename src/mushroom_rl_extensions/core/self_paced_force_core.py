import numpy as np
from mushroom_rl_extensions.core.self_paced_rl_core import SelfPacedRLCore


class SelfPacedForceCore(SelfPacedRLCore):
    """
    Enforces force curriculum budget during training
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_action_norm_data = []

    def _step(self, render):
        action = list()
        for idx_agent in range(len(self.agent)):
            action.append(self.agent[idx_agent].draw_action(self._state))

        # Force budget
        force_budget = self._state[-1]
        action[1] = np.clip(action[1], -force_budget, force_budget)
        self.force_action_norm_data.append(np.max(np.abs(action[1])))

        next_state, reward, absorbing, _ = self.mdp.step(action)

        pos = self.mdp.env.physics.position()

        self._episode_steps += 1

        if render:
            render_info = {"action": action, "reward": reward}
            self.mdp.render(render_info)

        last = not (self._episode_steps < self.mdp.info.horizon and not absorbing)

        state = self._state
        next_state = self._preprocess(next_state.copy())
        self._state = next_state

        return state, action, reward, next_state, absorbing, last
